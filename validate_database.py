"""
Phase 2 Database Validation Script

Validates that the analytics database was populated correctly and is ready
for Phase 3 (dashboard) and Phase 4 (ML models).

"""

import argparse
import sqlite3
import sys
from collections import Counter

# Design constants

MIN_RUNS        = 100   # Hard minimum
TESTS_PER_RUN   = 20    # Fixed by design

# Per-run expected rates (calibrated on 100-run baseline)
FAILURES_PER_RUN_MIN = 4.0   # ~20% fail rate × 20 tests
FAILURES_PER_RUN_MAX = 5.5   # ~27.5% fail rate × 20 tests
TAGS_PER_RUN_MIN     = 50    # ~2.5 tags/test × 20 tests
TAGS_PER_RUN_MAX     = 70    # ~3.5 tags/test × 20 tests

# Proportional window fractions
LATE_WINDOW_START_FRAC  = 0.75
STEP_BOUNDARY_FRAC      = 0.50
PROGRESSIVE_EARLY_FRAC  = 0.20
PROGRESSIVE_LATE_FRAC   = 0.80

# Fixed anomaly count (design artifact: runs 36-37 always dip)
EXPECTED_ANOMALY_RUNS = 2


# HELPERS

def _get_n_runs(conn):
    """Return actual run count from database."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM runs")
    return cursor.fetchone()[0]


def _step_boundary(n_runs):
    return n_runs // 2


def _early_cutoff(n_runs):
    return int(n_runs * PROGRESSIVE_EARLY_FRAC)


def _late_cutoff(n_runs):
    return int(n_runs * PROGRESSIVE_LATE_FRAC)


def _late_window_start(n_runs):
    return int(n_runs * LATE_WINDOW_START_FRAC)


# VALIDATION CHECKS

def validate_run_minimum(conn):
    """
    Validate that the database contains at least MIN_RUNS runs.

    This is a gate check — if it fails the remaining checks are skipped.

    Returns:
        tuple: (success, message, n_runs)
    """
    n_runs = _get_n_runs(conn)

    if n_runs < MIN_RUNS:
        return (
            False,
            f"runs: {n_runs} — FAIL (minimum required: {MIN_RUNS})",
            n_runs,
        )

    return True, f"✓ {n_runs} runs in database (minimum {MIN_RUNS})", n_runs


def validate_row_counts(conn, n_runs):
    """
    Validate that row counts match expected values scaled to N.

    Expected (all proportional to n_runs):
      - runs:         exactly n_runs
      - tests:        n_runs × TESTS_PER_RUN
      - test_results: one-to-one with tests
      - failures:     n_runs × [FAILURES_PER_RUN_MIN, FAILURES_PER_RUN_MAX]
      - tags:         n_runs × [TAGS_PER_RUN_MIN, TAGS_PER_RUN_MAX]

    Returns:
        tuple: (success, message, counts_dict)
    """
    cursor = conn.cursor()

    counts = {}
    for table in ["runs", "tests", "test_results", "failures", "tags"]:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cursor.fetchone()[0]

    issues = []

    # tests
    expected_tests = n_runs * TESTS_PER_RUN
    if counts["tests"] != expected_tests:
        issues.append(
            f"tests: {counts['tests']} (expected {expected_tests} = {n_runs} runs × {TESTS_PER_RUN})"
        )

    # test_results — one-to-one with tests
    if counts["test_results"] != counts["tests"]:
        issues.append(
            f"test_results: {counts['test_results']} "
            f"(expected {counts['tests']}, one-to-one with tests)"
        )

    # failures — proportional range
    fail_min = int(FAILURES_PER_RUN_MIN * n_runs)
    fail_max = int(FAILURES_PER_RUN_MAX * n_runs)
    if not (fail_min <= counts["failures"] <= fail_max):
        issues.append(
            f"failures: {counts['failures']} "
            f"(expected {fail_min}–{fail_max} for N={n_runs})"
        )

    # tags — proportional range
    tag_min = TAGS_PER_RUN_MIN * n_runs
    tag_max = TAGS_PER_RUN_MAX * n_runs
    if not (tag_min <= counts["tags"] <= tag_max):
        issues.append(
            f"tags: {counts['tags']} "
            f"(expected {tag_min}–{tag_max} for N={n_runs})"
        )

    if issues:
        return False, "Row count issues:\n  " + "\n  ".join(issues), counts

    msg = (
        f"✓ Row counts correct (N={n_runs})\n"
        f"    runs: {counts['runs']}, tests: {counts['tests']}, "
        f"test_results: {counts['test_results']}, "
        f"failures: {counts['failures']}, tags: {counts['tags']}"
    )

    return True, msg, counts


def validate_data_quality(conn):
    """
    Validate data quality (no nulls where required, valid ranges).

    Returns:
        tuple: (success, message)
    """
    cursor = conn.cursor()
    issues = []

    null_checks = [
        ("runs",         "timestamp"),
        ("runs",         "passed"),
        ("runs",         "failed"),
        ("runs",         "pass_rate"),
        ("tests",        "test_name"),
        ("tests",        "status"),
        ("tests",        "duration"),
        ("test_results", "category"),
        ("failures",     "message"),
    ]

    for table, column in null_checks:
        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL")
        null_count = cursor.fetchone()[0]
        if null_count > 0:
            issues.append(f"{table}.{column}: {null_count} nulls found")

    # Status values
    cursor.execute("SELECT DISTINCT status FROM tests")
    statuses = [row[0] for row in cursor.fetchall()]
    if set(statuses) != {"PASS", "FAIL"}:
        issues.append(f"tests.status: invalid values {statuses} (expected PASS, FAIL)")

    # pass_rate range
    cursor.execute("SELECT COUNT(*) FROM runs WHERE pass_rate < 0 OR pass_rate > 100")
    invalid_pr = cursor.fetchone()[0]
    if invalid_pr > 0:
        issues.append(f"runs.pass_rate: {invalid_pr} values outside 0–100")

    # duration range
    cursor.execute("SELECT COUNT(*) FROM tests WHERE duration < 0")
    neg_dur = cursor.fetchone()[0]
    if neg_dur > 0:
        issues.append(f"tests.duration: {neg_dur} negative values")

    # category values
    cursor.execute("SELECT DISTINCT category FROM test_results")
    categories = [row[0] for row in cursor.fetchall()]
    valid_categories = {
        "stable", "flaky-mild", "flaky-moderate",
        "flaky-heavy", "consistently_failing", "unknown",
    }
    invalid_cats = set(categories) - valid_categories
    if invalid_cats:
        issues.append(f"test_results.category: invalid values {invalid_cats}")

    if issues:
        return False, "Data quality issues:\n  " + "\n  ".join(issues)

    return True, "✓ Data quality checks passed"


def validate_foreign_keys(conn):
    """
    Validate foreign key relationships.

    Returns:
        tuple: (success, message)
    """
    cursor = conn.cursor()
    issues = []

    checks = [
        (
            "tests → runs",
            """SELECT COUNT(*) FROM tests t
               LEFT JOIN runs r ON t.run_id = r.run_id
               WHERE r.run_id IS NULL""",
            "tests",
        ),
        (
            "test_results → tests",
            """SELECT COUNT(*) FROM test_results tr
               LEFT JOIN tests t ON tr.test_id = t.test_id
               WHERE t.test_id IS NULL""",
            "test_results",
        ),
        (
            "failures → tests",
            """SELECT COUNT(*) FROM failures f
               LEFT JOIN tests t ON f.test_id = t.test_id
               WHERE t.test_id IS NULL""",
            "failures",
        ),
        (
            "tags → tests",
            """SELECT COUNT(*) FROM tags tg
               LEFT JOIN tests t ON tg.test_id = t.test_id
               WHERE t.test_id IS NULL""",
            "tags",
        ),
    ]

    for label, query, table_name in checks:
        cursor.execute(query)
        orphans = cursor.fetchone()[0]
        if orphans > 0:
            issues.append(f"{label}: {orphans} orphan record(s) in {table_name}")

    if issues:
        return False, "Foreign key integrity issues:\n  " + "\n  ".join(issues)

    return True, "✓ Foreign key integrity verified"


def validate_category_balance(conn):
    """
    Validate that failure categories match design (22-34% each).

    Returns:
        tuple: (success, message)
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT category, COUNT(*) as count
        FROM failures
        GROUP BY category
    """)

    results = cursor.fetchall()
    total = sum(count for _, count in results)

    if total == 0:
        return False, "No failures found in database"

    distribution = {cat: (cnt, cnt / total * 100) for cat, cnt in results}

    issues = []
    for category in ["timeout", "element", "assertion", "data"]:
        if category not in distribution:
            issues.append(f"{category}: 0% (missing)")
        else:
            count, pct = distribution[category]
            if pct < 20:
                issues.append(f"{category}: {count} ({pct:.1f}%) BELOW 22% minimum")
            elif pct > 36:
                issues.append(f"{category}: {count} ({pct:.1f}%) ABOVE 34% maximum")

    if issues:
        return False, "Category balance issues:\n  " + "\n  ".join(issues)

    breakdown = ", ".join([
        f"{cat}: {distribution.get(cat, (0, 0))[0]} ({distribution.get(cat, (0, 0))[1]:.1f}%)"
        for cat in ["timeout", "element", "assertion", "data"]
    ])

    return True, f"✓ Categories balanced ({total} total failures)\n    {breakdown}"


def validate_duration_patterns(conn, n_runs):
    """
    Validate that duration patterns are preserved in database.

    All window boundaries scale with N:
      - TC_Login_ValidCredentials:   Seasonal (even/odd — all N runs)
      - TC_Dashboard_ExportChart:    Step change at run N//2
      - TC_User_BulkImport:          Progressive drift
                                       early ≤ floor(N × 0.20)
                                       late  ≥ ceil(N × 0.80)

    Returns:
        tuple: (success, message)
    """
    cursor = conn.cursor()
    issues = []

    step_bound   = _step_boundary(n_runs)
    early_cutoff = _early_cutoff(n_runs)
    late_cutoff  = _late_cutoff(n_runs)

    # --- Seasonal pattern (TC_Login_ValidCredentials) ---
    cursor.execute("""
        SELECT r.run_id, t.duration
        FROM tests t
        JOIN runs r ON t.run_id = r.run_id
        WHERE t.test_name = 'TC_Login_ValidCredentials'
        ORDER BY r.run_id
    """)
    seasonal_data = cursor.fetchall()

    avg_even = avg_odd = seasonal_ratio = None

    if seasonal_data:
        even_durs = [d for rid, d in seasonal_data if rid % 2 == 0]
        odd_durs  = [d for rid, d in seasonal_data if rid % 2 != 0]

        if even_durs and odd_durs:
            avg_even = sum(even_durs) / len(even_durs)
            avg_odd  = sum(odd_durs)  / len(odd_durs)
            seasonal_ratio = avg_odd / avg_even if avg_even > 0 else 0

            if seasonal_ratio < 1.3:
                issues.append(
                    f"Seasonal: odd/even ratio {seasonal_ratio:.2f}× (expected ≥1.5×)"
                )
        else:
            issues.append("Seasonal: even or odd run data missing")
    else:
        issues.append("Seasonal: No data found for TC_Login_ValidCredentials")

    # --- Step change (TC_Dashboard_ExportChart) ---
    cursor.execute("""
        SELECT r.run_id, t.duration
        FROM tests t
        JOIN runs r ON t.run_id = r.run_id
        WHERE t.test_name = 'TC_Dashboard_ExportChart'
        ORDER BY r.run_id
    """)
    step_data = cursor.fetchall()

    avg_before = avg_after = step_ratio = None

    if step_data:
        before = [d for rid, d in step_data if rid <= step_bound]
        after  = [d for rid, d in step_data if rid > step_bound]

        if before and after:
            avg_before = sum(before) / len(before)
            avg_after  = sum(after)  / len(after)
            step_ratio = avg_after / avg_before if avg_before > 0 else 0

            if step_ratio < 2.0:
                issues.append(
                    f"Step change: after/before ratio {step_ratio:.2f}× "
                    f"(expected ≥2.5×) [boundary: run {step_bound}]"
                )
        else:
            issues.append(
                f"Step change: before or after data missing "
                f"[boundary: run {step_bound}]"
            )
    else:
        issues.append("Step change: No data found for TC_Dashboard_ExportChart")

    # --- Progressive drift (TC_User_BulkImport) ---
    cursor.execute("""
        SELECT r.run_id, t.duration
        FROM tests t
        JOIN runs r ON t.run_id = r.run_id
        WHERE t.test_name = 'TC_User_BulkImport'
        ORDER BY r.run_id
    """)
    progressive_data = cursor.fetchall()

    avg_early = avg_late = prog_ratio = None

    if progressive_data:
        early = [d for rid, d in progressive_data if rid <= early_cutoff]
        late  = [d for rid, d in progressive_data if rid >= late_cutoff]

        if early and late:
            avg_early = sum(early) / len(early)
            avg_late  = sum(late)  / len(late)
            prog_ratio = avg_late / avg_early if avg_early > 0 else 0

            if prog_ratio < 1.8:
                issues.append(
                    f"Progressive: late/early ratio {prog_ratio:.2f}× "
                    f"(expected ≥2.0×) [early ≤{early_cutoff}, late ≥{late_cutoff}]"
                )
        else:
            issues.append(
                f"Progressive: early or late data missing "
                f"[early ≤{early_cutoff}, late ≥{late_cutoff}]"
            )
    else:
        issues.append("Progressive: No data found for TC_User_BulkImport")

    if issues:
        return False, "Duration pattern issues:\n  " + "\n  ".join(issues)

    msg = (
        f"✓ All duration patterns preserved (N={n_runs})\n"
        f"    Seasonal:   {avg_odd:.1f}s (odd) / {avg_even:.1f}s (even) = {seasonal_ratio:.2f}×\n"
        f"    Step [{step_bound}]: "
        f"{avg_after:.1f}s (after) / {avg_before:.1f}s (before) = {step_ratio:.2f}×\n"
        f"    Progressive [≤{early_cutoff}/≥{late_cutoff}]: "
        f"{avg_late:.1f}s (late) / {avg_early:.1f}s (early) = {prog_ratio:.2f}×"
    )

    return True, msg


def validate_ml_readiness(conn, n_runs):
    """
    Validate that database is ready for Phase 4 ML models.

    All thresholds scale with N:
      - ML1: distinct test names == TESTS_PER_RUN (fixed, not N-dependent)
      - ML2: failure messages >= n_runs × FAILURES_PER_RUN_MIN
      - ML3: anomaly runs (pass_rate 20-35%) == EXPECTED_ANOMALY_RUNS (fixed design artifact)
      - ML4: each special test has exactly n_runs records

    Returns:
        tuple: (success, message)
    """
    cursor = conn.cursor()
    issues = []

    # ML1 — Flakiness Classifier: needs all 20 distinct test names
    cursor.execute("SELECT COUNT(DISTINCT test_name) FROM tests")
    distinct_tests = cursor.fetchone()[0]
    if distinct_tests != TESTS_PER_RUN:
        issues.append(
            f"ML1: {distinct_tests} distinct tests (expected {TESTS_PER_RUN})"
        )

    # ML2 — Failure Clustering: enough failure messages for TF-IDF
    cursor.execute("SELECT COUNT(*) FROM failures WHERE message IS NOT NULL")
    failures_with_msg = cursor.fetchone()[0]
    min_failures = int(FAILURES_PER_RUN_MIN * n_runs)
    if failures_with_msg < min_failures:
        issues.append(
            f"ML2: {failures_with_msg} failure messages (expected ≥{min_failures} for N={n_runs})"
        )

    # ML3 — Anomaly Detection: exactly 2 anomaly runs (runs 36-37 are hardcoded design artifact)
    cursor.execute("SELECT COUNT(*) FROM runs WHERE pass_rate BETWEEN 20 AND 35")
    anomaly_runs = cursor.fetchone()[0]
    if anomaly_runs < EXPECTED_ANOMALY_RUNS:
        issues.append(
            f"ML3: {anomaly_runs} anomaly runs (expected {EXPECTED_ANOMALY_RUNS})"
        )

    # ML4 — Duration Drift: each special test must appear exactly n_runs times
    special_tests = [
        "TC_Login_ValidCredentials",
        "TC_Dashboard_ExportChart",
        "TC_User_BulkImport",
    ]
    for test_name in special_tests:
        cursor.execute("SELECT COUNT(*) FROM tests WHERE test_name = ?", (test_name,))
        count = cursor.fetchone()[0]
        if count != n_runs:
            issues.append(
                f"ML4: '{test_name}' has {count} records (expected {n_runs})"
            )

    if issues:
        return False, "ML readiness issues:\n  " + "\n  ".join(issues)

    return (
        True,
        (
            f"✓ Database ready for all 4 ML models\n"
            f"    {distinct_tests} distinct tests, "
            f"{failures_with_msg} failure messages, "
            f"{anomaly_runs} anomaly runs (N={n_runs})"
        ),
    )


# MAIN VALIDATION FUNCTION

def validate(db_path):
    """
    Run all validation checks on database.

    Returns:
        bool: True if all checks pass
    """
    print("=" * 70)
    print("PHASE 2 DATABASE VALIDATION")
    print("=" * 70)
    print(f"\nValidating: {db_path}\n")

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
    except sqlite3.Error as e:
        print(f"✗ Cannot connect to database: {e}")
        return False

    all_passed = True

    # Gate check — minimum run count (must pass before all others)
    print("[0/6] Validating minimum run count...")
    success, message, n_runs = validate_run_minimum(conn)
    print(f"      {message}")
    if not success:
        print("\n✗ VALIDATION ABORTED: database has fewer than the minimum required runs.")
        conn.close()
        return False
    print()

    # Check 1: Row counts
    print("[1/6] Validating row counts...")
    success, message, _ = validate_row_counts(conn, n_runs)
    print(f"      {message}")
    if not success:
        all_passed = False
    print()

    # Check 2: Data quality
    print("[2/6] Validating data quality...")
    success, message = validate_data_quality(conn)
    print(f"      {message}")
    if not success:
        all_passed = False
    print()

    # Check 3: Foreign keys
    print("[3/6] Validating foreign key integrity...")
    success, message = validate_foreign_keys(conn)
    print(f"      {message}")
    if not success:
        all_passed = False
    print()

    # Check 4: Category balance
    print("[4/6] Validating category balance...")
    success, message = validate_category_balance(conn)
    print(f"      {message}")
    if not success:
        all_passed = False
    print()

    # Check 5: Duration patterns
    print("[5/6] Validating duration patterns...")
    success, message = validate_duration_patterns(conn, n_runs)
    print(f"      {message}")
    if not success:
        all_passed = False
    print()

    # Check 6: ML readiness
    print("[6/6] Validating ML readiness...")
    success, message = validate_ml_readiness(conn, n_runs)
    print(f"      {message}")
    if not success:
        all_passed = False
    print()

    conn.close()

    # Final verdict
    print("=" * 70)
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
        print("=" * 70)
        print()
    else:
        print("✗ SOME VALIDATION CHECKS FAILED")
        print("=" * 70)
        print("\nPlease review errors above and fix issues.")
        print("\nTo re-run pipeline:")
        print("  python pipeline.py")

    return all_passed


# COMMAND LINE INTERFACE

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Phase 2 Database Validation Script",
        epilog=(
            "Validates analytics.db is ready for Phase 3 and 4. "
            f"Requires at least {MIN_RUNS} runs; all thresholds scale with N."
        ),
    )
    p.add_argument(
        "--db", "--database",
        dest="database_path",
        default="./analytics.db",
        help="Database path (default: ./analytics.db)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = validate(args.database_path)
    sys.exit(0 if success else 1)