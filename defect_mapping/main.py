import argparse
import json
import os
import sqlite3
import sys
import pandas as pd
from pathlib import Path

from embedding_engine import EmbeddingEngine
from defect_mapper import DefectMapper

# Default Configuration
WEIGHTS = {
    "embedding_sim":     0.45,   # semantic similarity from sentence embeddings
    "tc_name_match":     0.25,   # test case name found in defect text
    "keyword_match":     0.10,   # failure keyword overlap
    "category_match":    0.10,   # failure category alignment
    "temporal_proximity": 0.10,  # date closeness
}

# Tester-to-Team Identity Map
TESTER_IDENTITY_MAP = {
    "sample.reporter@hpe.com": "TeamAlpha",
}

DEFAULT_ANALYTICS_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analytics.db")
DEFAULT_DEFECTS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defects.json")


def load_defects(json_path: str) -> list[dict]:
    """Load defects directly from the local JSON file."""
    if not os.path.exists(json_path):
        print(f"  [ERR] Defects file not found: {json_path}")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_failures_from_db(db_path: str) -> pd.DataFrame:
    """Load all FAIL test results from analytics.db."""
    if not Path(db_path).exists():
        print(f"  [ERR] Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        """
        SELECT
            tr.result_id,
            tr.run_id,
            r.team,
            r.suite_name,
            r.timestamp     AS run_timestamp,
            tr.test_name,
            tr.status,
            tr.duration_s,
            tr.failure_msg,
            tr.failure_kw,
            tr.tags
        FROM test_results tr
        JOIN runs r ON tr.run_id = r.run_id
        WHERE tr.status = 'FAIL'
        ORDER BY r.timestamp ASC, tr.test_name ASC
        """,
        conn,
    )
    conn.close()
    return df


def main():
    parser = argparse.ArgumentParser(description="Defect-to-Test-Run Mapping Script")
    parser.add_argument(
        "--db", 
        type=str, 
        default=DEFAULT_ANALYTICS_DB, 
        help="Path to the analytics.db file."
    )
    parser.add_argument(
        "--defects", 
        type=str, 
        default=DEFAULT_DEFECTS_JSON, 
        help="Path to the defects.json file."
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.40, 
        help="Minimum match score threshold (0.0 to 1.0)."
    )
    parser.add_argument(
        "--window", 
        type=int, 
        default=7, 
        help="Date window in days for pre-filter."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="all-MiniLM-L6-v2", 
        help="Sentence-transformer model name."
    )
    
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  Defect-to-Test-Run Mapping System")
    print("  Powered by Sentence-Transformer Embeddings (zero-shot)")
    print("=" * 70)
    print(f"  Configuration:")
    print(f"    DB Path:     {args.db}")
    print(f"    Defects:     {args.defects}")
    print(f"    Threshold:   {args.threshold}")
    print(f"    Window Days: {args.window}")
    print(f"    Model:       {args.model}")
    print("=" * 70)
    print()

    # Step 1: Load defects
    print("  Step 1: Loading defect data ...")
    defects = load_defects(args.defects)
    print(f"  [OK] {len(defects)} defects loaded")
    print()

    # Step 2: Load test failures
    print("  Step 2: Loading test failures from DB ...")
    failures_df = load_failures_from_db(args.db)
    if failures_df.empty:
        print("  [WARN] No test failures found in database.")
    else:
        print(f"  [OK] {len(failures_df)} failed test results loaded")
        print(f"    Unique tests : {failures_df['test_name'].nunique()}")
        print(f"    Date range   : {failures_df['run_timestamp'].min()} -> "
              f"{failures_df['run_timestamp'].max()}")
    print()

    # Step 3: Initialise embedding engine
    print("  Step 3: Initialising embedding engine ...")
    emb_engine = EmbeddingEngine(model_name=args.model)
    print()

    # Step 4: Run mapping
    print("  Step 4: Running mapping pipeline ...")
    mapper = DefectMapper(
        emb_engine=emb_engine,
        weights=WEIGHTS,
        identity_map=TESTER_IDENTITY_MAP,
        window_days=args.window,
        threshold=args.threshold,
    )
    
    if failures_df.empty:
        print("  [WARN] Cannot run mapping without failures.")
        mappings = pd.DataFrame()
    else:
        mappings = mapper.run_mapping(defects, failures_df)

    # Step 5: Generate and print report
    print("\n  Step 5: Generating Report ...")
    report = DefectMapper.generate_report(mappings, defects)
    print(report)

    print()
    print("  Done.")
    print()


if __name__ == "__main__":
    main()
