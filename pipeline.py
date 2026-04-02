"""
Phase 2 Data Ingestion Pipeline
================================

Parses every Robot Framework run folder produced by generate.py and loads the
data into analytics.db (Phase 2 schema).

What it populates
-----------------
  runs          — one row per CI run (from ci_metadata.json)
  tests         — one row per test per run (from output.xml)
  test_results  — one-to-one with tests; stores category / feature / priority
  failures      — one row per failed test; stores category + message text
  tags          — one row per tag per test per run
  ingestion_log — tracks which runs have been processed (idempotency)
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

# CONFIGURATION

DEFAULT_CONFIG = {
    "runs_dir":      "./runs",
    "database_path": "./analytics.db",
    "schema_path":   "./schema.sql",
    "batch_size":    50,   # commit every N runs for performance
    "force":         False,
}

# DATABASE INITIALISATION

def create_database(db_path: str, schema_path: str) -> sqlite3.Connection:
    """
    Create (or open) analytics.db and apply schema.sql.

    Using CREATE TABLE IF NOT EXISTS throughout the schema makes this safe to
    call on an existing database — tables that are already present are left
    untouched and no data is lost.

    Returns
    -------
    sqlite3.Connection
        Open connection with foreign keys enabled.
    """
    if not os.path.exists(schema_path):
        print(f"✗ Schema file not found: {schema_path}")
        print("  Ensure schema.sql is in the current directory or pass --schema <path>")
        sys.exit(1)

    with open(schema_path, "r", encoding="utf-8") as fh:
        schema_sql = fh.read()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        conn.executescript(schema_sql)
        conn.commit()
        print(f"✓ Database ready: {db_path}")
    except sqlite3.Error as exc:
        print(f"✗ Error applying schema: {exc}")
        conn.close()
        sys.exit(1)

    return conn


# TIMESTAMP HELPERS

def parse_rf_timestamp(ts_str: str) -> datetime:
    """
    Parse a Robot Framework timestamp string to a datetime object.

    RF format: ``YYYYMMDD HH:MM:SS.mmm``
    Example:   ``20241001 14:23:45.123``
    """
    try:
        return datetime.strptime(ts_str, "%Y%m%d %H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(ts_str, "%Y%m%d %H:%M:%S")


def calculate_duration(start_str: str, end_str: str) -> float:
    """
    Return elapsed seconds between two RF timestamp strings.

    Returns 0.0 on any parse error so the row is still inserted.
    """
    try:
        return (parse_rf_timestamp(end_str) - parse_rf_timestamp(start_str)).total_seconds()
    except Exception:
        return 0.0


# FAILURE MESSAGE CLASSIFICATION

_CATEGORY_PATTERNS = [
    ("timeout",     "still visible after",  "timeout"),
    ("element",     "not found after",      "retries"),
    ("assertion",   "Expected HTTP status", None),
    ("data",        "CSV export contained", "rows"),
    ("environment", "environment",          None),
    ("environment", "unreachable",          None),
]


def classify_failure_message(message: str) -> str:
    """
    Derive a failure category ('timeout', 'element', 'assertion', 'data',
    'environment') from a failure message string.

    Falls back to 'data' when no pattern matches so the row is still stored.
    """
    lower = message.lower()
    for category, primary, secondary in _CATEGORY_PATTERNS:
        if primary.lower() in lower:
            if secondary is None or secondary.lower() in lower:
                return category
    return "data"


def extract_failure_info(test_el) -> dict | None:
    """
    Extract failure category, message text, and failing keyword from a
    ``<test>`` XML element.

    Returns
    -------
    dict | None
        ``{'category': str, 'message': str, 'keyword_name': str | None}``
        or ``None`` when no failure message is present.
    """

    status_el = test_el.find("status")
    if status_el is None or status_el.get("status") != "FAIL":
        return None

    message = (status_el.text or "").strip()
    if not message:
        msg_el = test_el.find(".//msg[@level='FAIL']")
        message = (msg_el.text or "").strip() if msg_el is not None else ""

    if not message:
        return None

    # Walk innermost failing keyword
    keyword_name = None
    for kw_el in reversed(test_el.findall(".//kw")):
        kw_status = kw_el.find("status")
        if kw_status is not None and kw_status.get("status") == "FAIL":
            keyword_name = kw_el.get("name")
            break

    return {
        "category":     classify_failure_message(message),
        "message":      message,
        "keyword_name": keyword_name,
    }


# CONFIG LOOKUP

def get_test_category_from_config(test_name: str) -> tuple[str, float | None]:
    """
    Look up a test's (category, fail_probability) from config.py TESTS.

    Falls back to ('unknown', None) for tests not found in the config so the
    pipeline is tolerant of extra tests in the XML.

    Returns
    -------
    tuple[str, float | None]
        e.g. ``('flaky-moderate', 0.50)``
    """
    try:
        from config import TESTS
        for entry in TESTS:
            if entry[1] == test_name:   # index 1 = name
                return entry[4], entry[5]  # category, fail_prob
    except ImportError:
        pass
    return "unknown", None


# RUN FOLDER PARSER

def parse_run(run_folder: str, run_id: int) -> dict:
    """
    Parse one run folder and return all data ready for database insertion.

    Parameters
    ----------
    run_folder : str
        Path to a folder that contains ``output.xml`` and ``ci_metadata.json``.
    run_id : int
        Integer run identifier (extracted from the folder name).

    Returns
    -------
    dict
        ``{'run': {...}, 'tests': [...], 'failures': [...], 'tags': [...]}``

    """
    # ── metadata JSON ────────────────────────────────────────────────────────
    meta_path = os.path.join(run_folder, "ci_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    # Accept both key names produced by different generator versions
    pass_rate = meta.get("pass_rate_pct") if "pass_rate_pct" in meta else meta.get("pass_rate")
    if pass_rate is None:
        raise ValueError(
            f"ci_metadata.json in {run_folder} has neither 'pass_rate_pct' "
            f"nor 'pass_rate' key.  Available keys: {list(meta.keys())}"
        )

    run_data = {
        "run_id":       run_id,
        "build_number": meta["build_no"],
        "timestamp":    meta["timestamp"],
        "total_tests":  meta["total"],
        "passed":       meta["passed"],
        "failed":       meta["failed"],
        "pass_rate":    pass_rate,
        "environment":  meta.get("environment", "staging"),
        "executor":     meta.get("executor", "jenkins-agent-01"),
    }

    # ── XML ──────────────────────────────────────────────────────────────────
    xml_path = os.path.join(run_folder, "output.xml")
    root = ET.parse(xml_path).getroot()

    tests: list[dict]    = []
    failures: list[dict] = []
    tags: list[dict]     = []

    for test_el in root.findall(".//test"):
        test_name = test_el.get("name", "")
        status_el = test_el.find("status")
        if status_el is None:
            continue

        status     = status_el.get("status", "FAIL")
        start_time = status_el.get("starttime", "")
        end_time   = status_el.get("endtime", "")
        duration   = calculate_duration(start_time, end_time)

        category, fail_prob = get_test_category_from_config(test_name)

        # Extract feature / priority from tags
        feature  = "unknown"
        priority = "unknown"
        tag_names: list[str] = []

        for tag_el in test_el.findall("tag"):
            tag_text = (tag_el.text or "").strip()
            if not tag_text:
                continue
            tag_names.append(tag_text)
            if tag_text.startswith("feature_"):
                feature = tag_text
            elif tag_text.startswith("priority_"):
                priority = tag_text

        tests.append({
            "run_id":     run_id,
            "test_name":  test_name,
            "status":     status,
            "duration":   duration,
            "start_time": start_time,
            "end_time":   end_time,
            # Extra fields needed for test_results table
            "feature":          feature,
            "priority":         priority,
            "category":         category,
            "fail_probability": fail_prob,
        })

        for tag_name in tag_names:
            tags.append({"test_name": test_name, "tag_name": tag_name})

        if status == "FAIL":
            failure_info = extract_failure_info(test_el)
            if failure_info:
                failures.append({"test_name": test_name, **failure_info})

    return {"run": run_data, "tests": tests, "failures": failures, "tags": tags}


# DATABASE LOADING (single run, inside a transaction)

def load_run_data(conn: sqlite3.Connection, run_data: dict) -> dict:
    """
    Insert one run's data into all five tables inside a single transaction.

    The transaction is NOT committed here; the caller controls commit
    frequency (batch commits every N runs for performance).

    Parameters
    ----------
    conn : sqlite3.Connection
        Open database connection.
    run_data : dict
        As returned by ``parse_run()``.

    Returns
    -------
    dict
        ``{'tests_inserted': int, 'failures_inserted': int, 'tags_inserted': int}``

    """
    cursor = conn.cursor()

    try:
        # ── runs ─────────────────────────────────────────────────────────────
        cursor.execute(
            """
            INSERT INTO runs
                (run_id, build_number, timestamp, total_tests,
                 passed, failed, pass_rate, environment, executor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_data["run"]["run_id"],
                run_data["run"]["build_number"],
                run_data["run"]["timestamp"],
                run_data["run"]["total_tests"],
                run_data["run"]["passed"],
                run_data["run"]["failed"],
                run_data["run"]["pass_rate"],
                run_data["run"]["environment"],
                run_data["run"]["executor"],
            ),
        )

        # ── tests + test_results ─────────────────────────────────────────────
        # test_name → test_id mapping so failures/tags can reference it
        test_id_map: dict[str, int] = {}

        for test in run_data["tests"]:
            cursor.execute(
                """
                INSERT INTO tests
                    (run_id, test_name, status, duration, start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    test["run_id"],
                    test["test_name"],
                    test["status"],
                    test["duration"],
                    test["start_time"],
                    test["end_time"],
                ),
            )
            test_id = cursor.lastrowid
            if test_id is not None:
                test_id_map[test["test_name"]] = test_id

            cursor.execute(
                """
                INSERT INTO test_results
                    (test_id, feature, priority, category, fail_probability)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    test_id,
                    test["feature"],
                    test["priority"],
                    test["category"],
                    test["fail_probability"],
                ),
            )

        # ── failures ─────────────────────────────────────────────────────────
        failures_inserted = 0
        for failure in run_data["failures"]:
            test_id = test_id_map.get(failure["test_name"])
            if test_id is None:
                continue  # test not in this run (shouldn't happen)
            cursor.execute(
                """
                INSERT INTO failures (test_id, category, message, keyword_name)
                VALUES (?, ?, ?, ?)
                """,
                (test_id, failure["category"], failure["message"], failure["keyword_name"]),
            )
            failures_inserted += 1

        # ── tags ─────────────────────────────────────────────────────────────
        tags_inserted = 0
        for tag in run_data["tags"]:
            test_id = test_id_map.get(tag["test_name"])
            if test_id is None:
                continue
            cursor.execute(
                "INSERT INTO tags (test_id, tag_name) VALUES (?, ?)",
                (test_id, tag["tag_name"]),
            )
            tags_inserted += 1

        return {
            "tests_inserted":    len(run_data["tests"]),
            "failures_inserted": failures_inserted,
            "tags_inserted":     tags_inserted,
        }

    except sqlite3.Error as exc:
        conn.rollback()
        raise Exception(
            f"DB error while loading run {run_data['run']['run_id']}: {exc}"
        ) from exc


# INGESTION LOG HELPERS

def is_already_ingested(conn: sqlite3.Connection, run_id: int) -> bool:
    """Return True if run_id is in ingestion_log with status='success'."""
    row = conn.execute(
        "SELECT 1 FROM ingestion_log WHERE run_id = ? AND status = 'success'",
        (run_id,),
    ).fetchone()
    return row is not None


def log_ingestion_success(conn: sqlite3.Connection, run_id: int) -> None:
    """Record a successful ingestion in ingestion_log."""
    conn.execute(
        """
        INSERT OR REPLACE INTO ingestion_log (run_id, ingested_at, status, error_msg)
        VALUES (?, datetime('now'), 'success', NULL)
        """,
        (run_id,),
    )


def log_ingestion_error(conn: sqlite3.Connection, run_id: int, error_msg: str) -> None:
    """Record a failed ingestion in ingestion_log."""
    conn.execute(
        """
        INSERT OR REPLACE INTO ingestion_log (run_id, ingested_at, status, error_msg)
        VALUES (?, datetime('now'), 'error', ?)
        """,
        (run_id, error_msg[:2000]),  # cap length for display
    )
    conn.commit()  # error rows are committed immediately so they survive rollback


# MAIN PIPELINE

def run_pipeline(config: dict) -> dict:
    """
    Main pipeline execution:

    1. Validate input directory exists.
    2. Create / open database and apply schema.
    3. For each TeamAlpha_build_XXX folder (sorted):
       a. Skip if already in ingestion_log with status='success'.
       b. Parse output.xml + ci_metadata.json.
       c. Insert into runs / tests / test_results / failures / tags.
       d. Write success row to ingestion_log.
       e. Commit every batch_size runs.
    4. Print summary statistics.
    5. Verify row counts match expectations.
    """
    runs_dir    = config["runs_dir"]
    db_path     = config["database_path"]
    schema_path = config["schema_path"]
    batch_size  = config["batch_size"]
    force       = config.get("force", False)

    print("=" * 70)
    print()

    # ── Validate input directory ──────────────────────────────────────────────
    if not os.path.exists(runs_dir):
        print(f"✗ Input directory not found: {runs_dir}")
        sys.exit(1)

    folders = sorted(
        f for f in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, f))
        and f.startswith("TeamAlpha_build_")
    )

    if not folders:
        print(f"✗ No TeamAlpha_build_XXX folders found in {runs_dir}")
        sys.exit(1)

    print(f"  Input directory : {runs_dir}/")
    print(f"  Run folders     : {len(folders)}")
    print(f"  Database        : {db_path}")
    print(f"  Force re-ingest : {'yes' if force else 'no'}")
    print()

    # ── Connect / apply schema ────────────────────────────────────────────────
    conn = create_database(db_path, schema_path)
    print()

    # ── Process folders ───────────────────────────────────────────────────────
    stats = {
        "runs_processed":    0,
        "tests_inserted":    0,
        "failures_inserted": 0,
        "tags_inserted":     0,
        "runs_skipped":      0,
        "errors":            0,
    }

    print(f"Processing {len(folders)} folders...")
    print()

    for i, folder in enumerate(folders, 1):
        # Extract integer run_id from folder name (TeamAlpha_build_001 → 1)
        try:
            run_id = int(folder.rsplit("_", 1)[-1])
        except ValueError:
            print(f"  ⚠ Cannot parse run_id from folder name '{folder}' — skipping")
            stats["errors"] += 1
            continue

        folder_path = os.path.join(runs_dir, folder)

        # ── Idempotency check ─────────────────────────────────────────────────
        if not force and is_already_ingested(conn, run_id):
            stats["runs_skipped"] += 1
            continue

        # ── Parse ─────────────────────────────────────────────────────────────
        try:
            run_data = parse_run(folder_path, run_id)
        except Exception as exc:
            msg = str(exc)
            print(f"  ✗ Parse error  — {folder}: {msg[:120]}")
            log_ingestion_error(conn, run_id, f"parse: {msg}")
            stats["errors"] += 1
            continue

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            result = load_run_data(conn, run_data)
        except Exception as exc:
            msg = str(exc)
            print(f"  ✗ Load error   — {folder}: {msg[:120]}")
            log_ingestion_error(conn, run_id, f"load: {msg}")
            stats["errors"] += 1
            continue

        # ── Success ───────────────────────────────────────────────────────────
        log_ingestion_success(conn, run_id)

        stats["runs_processed"]    += 1
        stats["tests_inserted"]    += result["tests_inserted"]
        stats["failures_inserted"] += result["failures_inserted"]
        stats["tags_inserted"]     += result["tags_inserted"]

        # Batch commit
        if stats["runs_processed"] % batch_size == 0 or i == len(folders):
            conn.commit()
            print(
                f"  ✓  {stats['runs_processed']:3d}/{len(folders)} runs  "
                f"| {stats['tests_inserted']:5d} tests  "
                f"| {stats['failures_inserted']:4d} failures  "
                f"| {stats['tags_inserted']:5d} tags"
            )

    conn.commit()
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print()
    print(f"  Runs processed  : {stats['runs_processed']:4d}")
    print(f"  Runs skipped    : {stats['runs_skipped']:4d}  (already ingested)")
    print(f"  Tests inserted  : {stats['tests_inserted']:4d}")
    print(f"  Failures stored : {stats['failures_inserted']:4d}")
    print(f"  Tags stored     : {stats['tags_inserted']:4d}")
    if stats["errors"] > 0:
        print(f"  ✗ Errors        : {stats['errors']:4d}  (check output above)")
    print()

    # ── Row-count verification ────────────────────────────────────────────────
    print("Verifying database row counts...")
    checks = [
        ("runs",         stats["runs_processed"]    + stats["runs_skipped"], True),
        ("tests",        stats["tests_inserted"],                            False),
        ("test_results", stats["tests_inserted"],                            False),
        ("failures",     stats["failures_inserted"],                         False),
        ("tags",         stats["tags_inserted"],                             False),
    ]

    all_good = True
    cursor = conn.cursor()
    for table, expected, is_cumulative in checks:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        actual = cursor.fetchone()[0]
        note = " (cumulative total)" if is_cumulative else ""
        if actual == expected or (is_cumulative and actual >= expected):
            print(f"  ✓  {table:<14}: {actual:5d} rows{note}")
        else:
            print(f"  ✗  {table:<14}: {actual:5d} rows  (expected {expected}){note}")
            all_good = False

    print()
    if all_good:
        print("✓ Database verification passed")
    else:
        print("✗ Some row counts are unexpected — run validate_database.py for details")

    print()
    conn.close()
    return stats


# CLI

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Phase 2 — Data Ingestion Pipeline",
        epilog=(
            "Reads all TeamAlpha_build_XXX folders and loads them into analytics.db.\n"
            "The pipeline is idempotent: re-running it only processes new folders."
        ),
    )
    p.add_argument(
        "--runs-dir", default=DEFAULT_CONFIG["runs_dir"],
        help="Directory containing generated run folders (default: %(default)s)",
    )
    p.add_argument(
        "--db", "--database", dest="database_path",
        default=DEFAULT_CONFIG["database_path"],
        help="SQLite database path (default: %(default)s)",
    )
    p.add_argument(
        "--schema", dest="schema_path",
        default=DEFAULT_CONFIG["schema_path"],
        help="schema.sql path (default: %(default)s)",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"],
        help="Commit every N runs (default: %(default)s)",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Re-ingest all runs even if already in ingestion_log",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {
        "runs_dir":      args.runs_dir,
        "database_path": args.database_path,
        "schema_path":   args.schema_path,
        "batch_size":    args.batch_size,
        "force":         args.force,
    }
    result = run_pipeline(cfg)
    sys.exit(0 if result["errors"] == 0 else 1)