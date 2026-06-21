"""
Core defect-to-test-failure mapping engine.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from text_preprocess import (
    clean_defect_text,
    clean_failure_text,
    extract_tc_names,
    extract_failure_keyword,
    extract_sample_message,
    classify_failure_category,
)
from embedding_engine import EmbeddingEngine

def parse_defect_date(created_str: str) -> datetime:
    """Parse a JIRA-style ISO timestamp string into a datetime object."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(created_str, fmt).replace(tzinfo=None)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {created_str}")

class DefectMapper:
    def __init__(
        self,
        emb_engine: EmbeddingEngine,
        weights: dict,
        identity_map: dict,
        window_days: int = 7,
        threshold: float = 0.40,
    ):
        self.emb = emb_engine
        self.weights = weights
        self.identity_map = identity_map
        self.window_days = window_days
        self.threshold = threshold
    # ── Stage 1: Pre-Filter ─────────────────────────────────────────

    def _resolve_team(self, reporter_name: str) -> Optional[str]:
        """Look up the team for a defect reporter."""
        if not reporter_name:
            return None
        # Normalise email
        email = reporter_name.strip().lower()
        for pattern, team in self.identity_map.items():
            if pattern.lower() in email:
                return team
        return None

    def prefilter(
        self,
        defects: list[dict],
        failures_df: pd.DataFrame,
    ) -> list[tuple[dict, pd.DataFrame]]:
        """
        Apply pre-filters and return candidate pairs for each defect.

        For each defect, returns the subset of test failures that:
        - Are within ±window_days of the defect creation date
        - Come from the same team as the reporter (identity match)

        Returns
        -------
        list of (defect_dict, matching_failures_df) tuples
        """
        candidates = []

        for defect in defects:
            defect_date = parse_defect_date(defect["created"])
            reporter = defect.get("reporter_name", "")
            team = self._resolve_team(reporter)

            # Date window filter
            window_start = defect_date - timedelta(days=self.window_days)
            window_end = defect_date + timedelta(days=self.window_days)

            mask = pd.Series([True] * len(failures_df), index=failures_df.index)

            # Apply date filter if timestamps are available
            if "run_timestamp" in failures_df.columns:
                ts = pd.to_datetime(failures_df["run_timestamp"], errors="coerce")
                mask &= (ts >= window_start) & (ts <= window_end)

            # Apply team filter if identity is resolved
            if team and "team" in failures_df.columns:
                mask &= failures_df["team"] == team

            filtered = failures_df[mask]

            if not filtered.empty:
                candidates.append((defect, filtered))
            else:
                # If no candidates after filtering, include with note
                candidates.append((defect, pd.DataFrame()))

        return candidates

    # ── Stage 2: Scoring ────────────────────────────────────────────

    def _score_tc_name(self, defect: dict, test_name: str) -> float:
        """Check if the test case name appears in the defect text."""
        text = f"{defect.get('summary', '')} {defect.get('description', '')}"
        tc_names = extract_tc_names(text)
        if test_name in tc_names:
            return 1.0
        # Partial match — check if any extracted TC name is a substring
        for tc in tc_names:
            if tc in test_name or test_name in tc:
                return 0.8
        return 0.0

    def _score_keyword(self, defect: dict, failure_kw: Optional[str]) -> float:
        """Compare defect's failure keyword with the test failure keyword."""
        if not failure_kw:
            return 0.0

        defect_kw = extract_failure_keyword(defect.get("description", ""))
        if not defect_kw:
            return 0.0

        # Exact match
        if defect_kw.lower() == failure_kw.lower():
            return 1.0

        # Token overlap
        defect_tokens = set(defect_kw.lower().split())
        failure_tokens = set(failure_kw.lower().split())
        if not defect_tokens or not failure_tokens:
            return 0.0

        overlap = len(defect_tokens & failure_tokens)
        union = len(defect_tokens | failure_tokens)
        return overlap / union if union > 0 else 0.0

    def _score_category(self, defect: dict, failure_msg: Optional[str]) -> float:
        """Check if defect and failure classify into the same category."""
        defect_text = f"{defect.get('summary', '')} {defect.get('description', '')}"
        defect_cat = classify_failure_category(defect_text)
        failure_cat = classify_failure_category(failure_msg or "")

        if defect_cat == "unknown" or failure_cat == "unknown":
            return 0.0
        return 1.0 if defect_cat == failure_cat else 0.0

    def _score_temporal(self, defect: dict, run_timestamp: str) -> float:
        """Score based on date proximity (1.0 = same day, 0.0 = 7+ days)."""
        try:
            defect_date = parse_defect_date(defect["created"])
            run_date = pd.to_datetime(run_timestamp).to_pydatetime()
            if run_date.tzinfo:
                run_date = run_date.replace(tzinfo=None)
            days_apart = abs((defect_date - run_date).total_seconds()) / 86400.0
            return max(0.0, 1.0 - (days_apart / self.window_days))
        except Exception:
            return 0.0

    def score_pair(self, defect: dict, failure_row: pd.Series) -> dict:
        """
        Compute the weighted match score for a single (defect, failure) pair.

        Returns
        -------
        dict with keys:
            score           — final weighted score (0.0 to 1.0)
            embedding_sim   — cosine similarity from embeddings
            tc_name_match   — TC name match score
            keyword_match   — keyword overlap score
            category_match  — category alignment score
            temporal_prox   — temporal proximity score
        """
        # Prepare texts for embedding
        defect_text = clean_defect_text(defect)
        failure_text = clean_failure_text(
            failure_row.get("failure_msg"),
            failure_row.get("failure_kw"),
        )

        # 1. Embedding cosine similarity
        if failure_text:
            emb_sim = self.emb.similarity(defect_text, failure_text)
        else:
            emb_sim = 0.0

        # 2. TC name match
        tc_score = self._score_tc_name(defect, failure_row.get("test_name", ""))

        # 3. Keyword match
        kw_score = self._score_keyword(defect, failure_row.get("failure_kw"))

        # 4. Category match
        cat_score = self._score_category(defect, failure_row.get("failure_msg"))

        # 5. Temporal proximity
        temporal = self._score_temporal(
            defect, failure_row.get("run_timestamp", "")
        )

        # Weighted combination
        final_score = (
            self.weights["embedding_sim"]      * emb_sim
            + self.weights["tc_name_match"]    * tc_score
            + self.weights["keyword_match"]    * kw_score
            + self.weights["category_match"]   * cat_score
            + self.weights["temporal_proximity"] * temporal
        )

        return {
            "score":          round(final_score, 4),
            "embedding_sim":  round(emb_sim, 4),
            "tc_name_match":  round(tc_score, 4),
            "keyword_match":  round(kw_score, 4),
            "category_match": round(cat_score, 4),
            "temporal_prox":  round(temporal, 4),
        }

    # ── Stage 3: Full Pipeline ──────────────────────────────────────

    def run_mapping(
        self,
        defects: list[dict],
        failures_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Run the full defect-to-failure mapping pipeline.

        Parameters
        ----------
        defects : list[dict]
            JIRA defect dicts.
        failures_df : pd.DataFrame
            Test failure records (status == 'FAIL') with columns:
            result_id, run_id, test_name, failure_msg, failure_kw,
            run_timestamp, team, etc.

        Returns
        -------
        pd.DataFrame
            Mapping results with columns:
            defect_key, result_id, run_id, test_name, score,
            embedding_sim, tc_name_match, keyword_match,
            category_match, temporal_prox, defect_summary
        """
        print(f"\n{'='*70}")
        print("  Defect-to-Test-Run Mapping Engine")
        print(f"{'='*70}\n")
        print(f"  Defects to map    : {len(defects)}")
        print(f"  Test failures     : {len(failures_df)}")
        print(f"  Date window       : +/-{self.window_days} days")
        print(f"  Score threshold   : {self.threshold}")
        print()

        # Stage 1: Pre-filter
        print("  Stage 1: Pre-filtering candidates ...")
        candidate_pairs = self.prefilter(defects, failures_df)
        total_candidates = sum(len(f) for _, f in candidate_pairs)
        print(f"  [OK] {total_candidates} candidate pairs generated\n")

        # Stage 2: Score each candidate pair
        print("  Stage 2: Computing semantic scores ...")
        all_mappings = []

        for defect, filtered_failures in candidate_pairs:
            defect_key = defect["key"]

            if filtered_failures.empty:
                print(f"    [WARN] {defect_key}: No candidates after pre-filter")
                continue

            # Check for consolidated defects (multiple TC names)
            defect_text = f"{defect.get('summary', '')} {defect.get('description', '')}"
            tc_names_in_defect = extract_tc_names(defect_text)
            is_consolidated = len(tc_names_in_defect) > 1

            pair_scores = []
            for _, failure_row in filtered_failures.iterrows():
                scores = self.score_pair(defect, failure_row)
                scores["defect_key"] = defect_key
                scores["defect_summary"] = defect.get("summary", "")[:100]
                scores["result_id"] = failure_row.get("result_id", "")
                scores["run_id"] = failure_row.get("run_id", "")
                scores["test_name"] = failure_row.get("test_name", "")
                scores["defect_status"] = defect.get("status", "")
                scores["defect_priority"] = defect.get("priority", "")
                scores["defect_project"] = defect.get("project", "")
                pair_scores.append(scores)

            if not pair_scores:
                continue

            pair_df = pd.DataFrame(pair_scores)

            # Stage 3: Select best matches
            above_threshold = pair_df[pair_df["score"] >= self.threshold]

            if above_threshold.empty:
                # Take the single best even if below threshold, but flag it
                best = pair_df.nlargest(1, "score")
                best = best.copy()
                best["below_threshold"] = True
                all_mappings.append(best)
                print(
                    f"    [WARN] {defect_key}: Best score {best.iloc[0]['score']:.3f} "
                    f"< threshold {self.threshold}"
                )
            elif is_consolidated:
                # For consolidated defects: pick best match per TC name
                selected = []
                for tc_name in tc_names_in_defect:
                    tc_matches = above_threshold[
                        above_threshold["test_name"] == tc_name
                    ]
                    if not tc_matches.empty:
                        best_for_tc = tc_matches.nlargest(1, "score")
                        best_for_tc = best_for_tc.copy()
                        best_for_tc["below_threshold"] = False
                        selected.append(best_for_tc)

                if selected:
                    all_mappings.extend(selected)
                    print(
                        f"    [OK] {defect_key}: Consolidated -> "
                        f"{len(selected)} test(s) matched"
                    )
                else:
                    # Fallback: best overall match
                    best = above_threshold.nlargest(1, "score")
                    best = best.copy()
                    best["below_threshold"] = False
                    all_mappings.append(best)
                    print(
                        f"    [OK] {defect_key}: Score={best.iloc[0]['score']:.3f}"
                    )
            else:
                # Standard: pick the single best match
                best = above_threshold.nlargest(1, "score")
                best = best.copy()
                best["below_threshold"] = False
                all_mappings.append(best)
                print(
                    f"    [OK] {defect_key}: -> {best.iloc[0]['test_name']} "
                    f"(score={best.iloc[0]['score']:.3f})"
                )

        if not all_mappings:
            print("\n  [ERR] No mappings found.")
            return pd.DataFrame()

        result = pd.concat(all_mappings, ignore_index=True)
        result = result.sort_values("score", ascending=False).reset_index(drop=True)

        print(f"\n  [OK] {len(result)} total mappings generated")
        print(f"{'='*70}\n")

        return result

    # ── Report Generation ───────────────────────────────────────────

    @staticmethod
    def generate_report(mappings: pd.DataFrame, defects: list[dict]) -> str:
        """
        Generate a human-readable console report of the mapping results.
        """
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  DEFECT-TO-TEST-RUN MAPPING REPORT")
        lines.append("=" * 70)
        lines.append("")

        if mappings.empty:
            lines.append("  No mappings were generated.")
            return "\n".join(lines)

        # Summary stats
        total_defects = len(defects)
        mapped_defects = mappings["defect_key"].nunique()
        below_threshold = (
            mappings["below_threshold"].sum() if "below_threshold" in mappings.columns else 0
        )
        avg_score = mappings["score"].mean()
        max_score = mappings["score"].max()
        min_score = mappings["score"].min()

        lines.append("  -- Summary -----------------------------------------")
        lines.append(f"  Total defects       : {total_defects}")
        lines.append(f"  Mapped defects      : {mapped_defects}")
        lines.append(
            f"  Coverage            : {mapped_defects / total_defects * 100:.1f}%"
        )
        lines.append(f"  Total mappings      : {len(mappings)}")
        lines.append(f"  Below threshold     : {int(below_threshold)}")
        lines.append(f"  Avg score           : {avg_score:.3f}")
        lines.append(f"  Score range         : [{min_score:.3f} — {max_score:.3f}]")
        lines.append("")

        # Per-defect details
        lines.append("  -- Per-Defect Mappings -----------------------------")
        lines.append("")

        for defect_key in mappings["defect_key"].unique():
            defect_maps = mappings[mappings["defect_key"] == defect_key]
            summary = defect_maps.iloc[0].get("defect_summary", "")
            status = defect_maps.iloc[0].get("defect_status", "")
            priority = defect_maps.iloc[0].get("defect_priority", "")
            project = defect_maps.iloc[0].get("defect_project", "")

            lines.append(f"  [{defect_key}] {summary}")
            lines.append(f"  Status: {status} | Priority: {priority} | Project: {project}")
            lines.append("")

            for _, row in defect_maps.iterrows():
                below = " [!] BELOW THRESHOLD" if row.get("below_threshold") else ""
                lines.append(
                    f"    -> {row['test_name']:40s}  "
                    f"Score: {row['score']:.3f}{below}"
                )
                lines.append(
                    f"      Embedding: {row['embedding_sim']:.3f}  "
                    f"TC-Name: {row['tc_name_match']:.1f}  "
                    f"Keyword: {row['keyword_match']:.2f}  "
                    f"Category: {row['category_match']:.1f}  "
                    f"Temporal: {row['temporal_prox']:.2f}"
                )
                lines.append(f"      Run: {row['run_id']}")
                lines.append("")

            lines.append("  " + "-" * 50)
            lines.append("")

        # Unmapped defects
        mapped_keys = set(mappings["defect_key"].unique())
        unmapped = [d for d in defects if d["key"] not in mapped_keys]
        if unmapped:
            lines.append("  -- Unmapped Defects -------------------------------")
            for d in unmapped:
                lines.append(f"    [X] [{d['key']}] {d['summary'][:80]}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)
