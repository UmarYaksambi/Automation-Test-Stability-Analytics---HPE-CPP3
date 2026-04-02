"""
Phase 1 Synthetic Test Data Generator

Generates 100 Robot Framework output.xml files implementing the design decisions
from design_doc.md:
  - Design Question 1: Class balance with varied failure probabilities
  - Design Question 2: Category balance with 70/30 primary/secondary split  
  - Design Question 3: Duration patterns (seasonal, step-change, progressive)

Output Structure:
  runs/
    TeamAlpha_build_001/
      output.xml          ← Robot Framework XML format
      ci_metadata.json    ← Build metadata (pass rate, counts, timestamp)
    TeamAlpha_build_002/
      ...
    TeamAlpha_build_100/
      ...

Usage:
  python generate.py                    # Generate 100 runs with defaults
  python generate.py --num-runs 50      # Generate 50 runs
  python generate.py --seed 123         # Use different random seed
  python generate.py --output-dir ./my_runs  # Custom output directory

"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

from config import DEFAULT_CONFIG, TESTS, DEPENDENCIES

# FAILURE MESSAGE GENERATORS

def gen_timeout_msg(rng):
    """
    Generate timeout failure message.
    
    Pattern: Element '{element}' still visible after {seconds}s timeout
    
    Root cause: Slow backend, network latency, page load issues
    Who fixes: Infrastructure team, DevOps
    """
    elements = [
        "loading-spinner",
        "overlay-modal", 
        "progress-bar",
        "auth-redirect",
        "session-token"
    ]
    timeouts = [15, 20, 30, 45]
    return f"Element '{rng.choice(elements)}' still visible after {rng.choice(timeouts)}s timeout"


def gen_element_msg(rng):
    """
    Generate element-not-found failure message.
    
    Pattern: Element with locator '{locator}' not found after {retries} retries
    
    Root cause: DOM structure changed, timing issue, element not rendered
    Who fixes: Frontend developers
    """
    locators = [
        "id=widget-container",
        "id=submit-btn",
        "css=.data-grid",
        "id=modal-confirm",
        "css=.nav-item"
    ]
    retries = [3, 5, 7]
    return f"Element with locator '{rng.choice(locators)}' not found after {rng.choice(retries)} retries"


def gen_assertion_msg(rng):
    """
    Generate assertion failure message.
    
    Pattern: Expected HTTP status '{expected}' but got '{actual}' — {description}
    
    Root cause: API returned wrong status, logic error, validation failure
    Who fixes: Backend developers, API team
    """
    pairs = [
        ("200", "500", "Internal Server Error"),
        ("200", "404", "Not Found"),
        ("201", "400", "Bad Request"),
        ("200", "503", "Service Unavailable")
    ]
    exp, got, desc = rng.choice(pairs)
    return f"Expected HTTP status '{exp}' but got '{got}' — {desc}"


def gen_data_msg(rng):
    """
    Generate data-missing failure message.
    
    Pattern: CSV export contained {rows} rows — expected at least {min_rows} records for {date_range}
    
    Root cause: Database query issue, ETL pipeline broken, empty result set
    Who fixes: Data engineers, Database team
    """
    rows = [0, 1, 2]
    mins = [50, 100, 200]
    ranges = ["Oct 2024", "last 30 days", "Q4 2024", "last 7 days"]
    return f"CSV export contained {rng.choice(rows)} rows — expected at least {rng.choice(mins)} records for {rng.choice(ranges)}"


def gen_environment_msg(rng):
    """
    Generate environment failure message.
    
    Used only for stable tests that fail (environment issues, not test issues)
    """
    messages = [
        "Connection refused — test environment unreachable",
        "Timeout waiting for application server to respond",
        "Suite setup failed — environment health check failed",
        "Unable to launch browser — infrastructure error",
    ]
    return rng.choice(messages)


# Map failure type to generator function
FAIL_GEN = {
    "timeout":     gen_timeout_msg,
    "element":     gen_element_msg,
    "assertion":   gen_assertion_msg,
    "data":        gen_data_msg,
    "environment": gen_environment_msg
}

# Map failure type to Robot Framework keyword name (for inner <kw> element)
FAIL_KW = {
    "timeout":     "Wait Until Element Is Visible",
    "element":     "Click Element",
    "assertion":   "Should Be Equal As Integers",
    "data":        "Should Not Be Empty",
    "environment": "Environment_Setup"
}

# PASS RATE CURVE

def run_pass_rate(n, num_runs, anomaly_runs, anomaly_pass_rate):
    """
    Return the suite-level pass-rate target for run n (1-indexed).

    Phase boundaries scale proportionally with num_runs so the curve shape
    is preserved for any N >= 100.  Anomaly runs (36-37) remain fixed build
    numbers — they are a design artifact, not a sliding window.

    For num_runs=100 the boundaries are identical to the original design:
      - Phase 1 (1-25):   70-80%  (early instability)
      - Phase 2 (26-35):  65-72%  (gradual decline)
      - Runs 36-37:       ~27%    (ANOMALY — fixed build numbers)
      - Phase 3 (36-45):  60-65%  (partial recovery)
      - Phase 4 (46-75):  65-80%  (recovery sprint)
      - Phase 5 (76-N):   82-95%  (stable high quality)

    Args:
        n:                 Run number (1-indexed)
        num_runs:          Total number of runs being generated
        anomaly_runs:      List of anomaly run numbers (e.g., [36, 37])
        anomaly_pass_rate: Pass rate during anomaly (e.g., 0.27)

    Returns:
        float: Target pass rate for this run (0.0 to 1.0)
    """
    # Anomaly runs override everything (fixed design artifact)
    if n in anomaly_runs:
        return anomaly_pass_rate

    # Compute proportional phase boundaries
    p25 = int(num_runs * 0.25)   # Phase 1 end  — 25 for N=100, 50 for N=200
    p35 = int(num_runs * 0.35)   # Phase 2 end  — 35 for N=100, 70 for N=200
    p45 = int(num_runs * 0.45)   # Phase 3 end  — 45 for N=100, 90 for N=200
    p75 = int(num_runs * 0.75)   # Phase 4 end  — 75 for N=100, 150 for N=200

    # Phase 1: Early instability
    if n <= p25:
        return random.uniform(0.70, 0.80)

    # Phase 2: Gradual decline
    elif n <= p35:
        return random.uniform(0.65, 0.72)

    # Phase 3: Partial recovery (anomaly runs are already handled above)
    elif n <= p45:
        return random.uniform(0.60, 0.65)

    # Phase 4: Recovery sprint
    elif n <= p75:
        return random.uniform(0.65, 0.80)

    # Phase 5: Stable high quality
    else:
        return random.uniform(0.82, 0.95)


# DURATION PATTERNS

def base_duration(test_name, n, num_runs, rng):
    """
    Calculate base test duration based on test name and run number.

    All pattern boundaries scale proportionally with num_runs so the shape
    is preserved for any N >= 100.  For num_runs=100 the boundaries are
    identical to the original design.

      1. TC_Login_ValidCredentials:  Seasonal (even/odd — all runs, no boundary)
      2. TC_Dashboard_ExportChart:   Step change at run N//2  (50 for N=100)
      3. TC_User_BulkImport:         Progressive drift over 4 phases

    Args:
        test_name: Name of the test
        n:         Run number (1-indexed)
        num_runs:  Total number of runs being generated
        rng:       Random number generator

    Returns:
        float: Base duration in seconds (before failure overhead)
    """
    # Pattern 1: Seasonal (alternating even/odd — no boundary needed)
    if test_name == "TC_Login_ValidCredentials":
        if n % 2 == 0:
            return rng.uniform(2.0, 3.5)   # Even: fast server
        else:
            return rng.uniform(4.5, 6.5)   # Odd:  slow server

    # Pattern 2: Step change at mid-point (scales with N)
    if test_name == "TC_Dashboard_ExportChart":
        step_boundary = num_runs // 2       # 50 for N=100, 100 for N=200
        if n <= step_boundary:
            return rng.uniform(3.0, 5.0)   # Before: normal performance
        else:
            return rng.uniform(12.0, 15.0) # After:  ~3× slower post-deployment

    # Pattern 3: Progressive drift (4 phases, all boundaries scale with N)
    if test_name == "TC_User_BulkImport":
        p40 = int(num_runs * 0.40)   # 40 for N=100, 80  for N=200
        p50 = int(num_runs * 0.50)   # 50 for N=100, 100 for N=200
        p65 = int(num_runs * 0.65)   # 65 for N=100, 130 for N=200

        if n <= p40:
            return rng.uniform(10.0, 12.0)  # Phase 1: normal baseline
        elif n <= p50:
            return rng.uniform(14.0, 18.0)  # Phase 2a: starting to slow
        elif n <= p65:
            return rng.uniform(18.0, 24.0)  # Phase 2b: noticeably slower
        else:
            return rng.uniform(28.0, 45.0)  # Phase 3: significantly degraded

    # All other tests: normal random variation
    return rng.uniform(1.2, 8.5)


def test_duration(test_name, n, num_runs, status, rng):
    """
    Calculate total test duration including failure overhead.

    Failed tests add 5-15 seconds for timeout/retry overhead.

    Args:
        test_name: Name of the test
        n:         Run number (1-indexed)
        num_runs:  Total number of runs being generated
        status:    Test status ("PASS" or "FAIL")
        rng:       Random number generator

    Returns:
        float: Total duration in seconds (rounded to 3 decimal places)
    """
    d = base_duration(test_name, n, num_runs, rng)

    # Failed tests have additional overhead for timeouts, retries, error handling
    if status == "FAIL":
        d += rng.uniform(5.0, 15.0)

    return round(d, 3)


# TIMESTAMP HELPERS

def fmt_ts(dt):
    """
    Format datetime as Robot Framework timestamp string.
    
    Format: YYYYMMDD HH:MM:SS.mmm
    Example: 20241001 14:23:45.123
    
    Args:
        dt: datetime object
    
    Returns:
        str: Formatted timestamp
    """
    return dt.strftime("%Y%m%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


# TEST OUTCOME DECISION

def decide_outcome(category, fail_prob, is_anomaly, rng):
    """
    Determine if a test should pass or fail.
    
    Logic varies by test category:
      - stable: Always passes (unless anomaly)
      - flaky-*: Uses own fail_prob (independent of suite curve)
      - consistently_failing: Uses own high fail_prob
    
    During anomaly runs, all categories get worse:
      - stable: 80% fail rate (normally 0%)
      - flaky: fail_prob increases by +30%
      - consistently_failing: fail_prob increases by +15%
    
    Args:
        category: Test category (stable, flaky-mild, etc.)
        fail_prob: Base failure probability (0.0 to 1.0)
        is_anomaly: Whether this is an anomaly run
        rng: Random number generator
    
    Returns:
        bool: True if test passes, False if test fails
    """
    # Anomaly runs: Everything gets worse
    if is_anomaly:
        if category == "stable":
            # Stable tests fail 80% of the time during anomaly
            return rng.random() > 0.80
        
        if category in ("flaky-mild", "flaky-moderate", "flaky-heavy"):
            # Flaky tests get +30% worse, capped at 95%
            return rng.random() > min(fail_prob + 0.30, 0.95)
        
        if category == "consistently_failing":
            # Consistently-failing get +15% worse, capped at 99%
            return rng.random() > min(fail_prob + 0.15, 0.99)
        
        return True  # Fallback: pass
    
    # Normal runs: Use designed probabilities
    if category == "stable":
        # Stable tests always pass (they fail via suite curve, handled separately)
        return True
    
    if category in ("flaky-mild", "flaky-moderate", "flaky-heavy"):
        # Flaky tests use their OWN probability (ignore suite curve)
        # This is CRITICAL: don't use suite pass rate for flaky tests
        return rng.random() > fail_prob
    
    if category == "consistently_failing":
        # Consistently-failing tests use their own high probability
        return rng.random() > fail_prob
    
    return True  # Fallback: pass


# XML GENERATION

def _indent(elem, level=0):
    """
    Add pretty-print whitespace to an ElementTree element in-place.
    
    This makes the output XML human-readable with proper indentation.
    
    Args:
        elem: ElementTree Element to indent
        level: Current indentation level (used recursively)
    """
    pad = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = pad + "  "
        if not child.tail or not child.tail.strip():
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def build_test_xml(test, passed, n, num_runs, is_anomaly, current_dt, rng, force_fail=False, force_pass=False):
    """
    Build XML element for a single test case.

    Creates Robot Framework <test> element with:
      - Test metadata (id, name, tags)
      - Keyword element with execution details
      - Status element with timing and result
      - Failure messages (if test failed)

    Args:
        test:        Test tuple from TESTS configuration
        passed:      Whether test naturally passed (before forcing)
        n:           Run number (1-indexed)
        num_runs:    Total number of runs being generated
        is_anomaly:  Whether this is an anomaly run
        current_dt:  Current datetime (start of this test)
        rng:         Random number generator
        force_fail:  Override to force failure (for pass rate correction)
        force_pass:  Override to force pass (for pass rate correction)

    Returns:
        tuple: (test_element, status, next_datetime)
            - test_element:  ET.Element for the test
            - status:        "PASS" or "FAIL"
            - next_datetime: Datetime after this test completes
    """
    # Unpack test configuration
    tid, name, feature_tag, priority_tag, category, fail_prob, _, primary, secondary, prim_prob = test
    
    # Determine final status (with forcing overrides for pass rate correction)
    if force_fail:
        status = "FAIL"
    elif force_pass:
        status = "PASS"
    else:
        status = "PASS" if passed else "FAIL"
    
    # Calculate timing
    dur = test_duration(name, n, num_runs, status, rng)
    start_dt = current_dt
    end_dt = start_dt + timedelta(seconds=dur)
    
    start_s = fmt_ts(start_dt)
    end_s = fmt_ts(end_dt)
    info_ts = fmt_ts(start_dt + timedelta(milliseconds=200))
    
    # Build test element
    test_el = ET.Element("test")
    test_el.set("id", tid)
    test_el.set("name", name)
    
    # Add tags
    ET.SubElement(test_el, "tag").text = "alpha_regression"
    ET.SubElement(test_el, "tag").text = feature_tag
    ET.SubElement(test_el, "tag").text = priority_tag
    
    # Add main keyword
    kw = ET.SubElement(test_el, "kw")
    kw.set("name", "Run Test Steps")
    kw.set("library", "SeleniumLibrary")
    
    # Info message
    info_msg = ET.SubElement(kw, "msg")
    info_msg.set("timestamp", info_ts)
    info_msg.set("level", "INFO")
    info_msg.text = f"Executing {name}"
    
    # Handle pass/fail cases
    if status == "PASS":
        # Passing test: Just add status to keyword and test
        kw_status = ET.SubElement(kw, "status")
        kw_status.set("status", "PASS")
        kw_status.set("starttime", start_s)
        kw_status.set("endtime", end_s)
        
        outer_status = ET.SubElement(test_el, "status")
        outer_status.set("status", "PASS")
        outer_status.set("starttime", start_s)
        outer_status.set("endtime", end_s)
    
    else:
        # Failing test: Determine failure type and add failure details
        
        # Determine failure type (primary vs secondary)
        if category == "stable":
            # Stable tests fail due to environment issues, not test issues
            ftype = "environment"
        elif prim_prob is None:
            # No primary/secondary split (shouldn't happen, but handle gracefully)
            ftype = primary or "timeout"
        else:
            # Use primary/secondary split based on probability
            # CRITICAL: Use < not <= to get exact 70/30 split
            ftype = primary if rng.random() < prim_prob else secondary
        
        # Generate failure message
        fail_msg = FAIL_GEN[ftype](rng)
        fail_ts = fmt_ts(end_dt - timedelta(milliseconds=rng.randint(5, 50)))
        inner_kw_name = FAIL_KW[ftype]
        inner_start = fmt_ts(end_dt - timedelta(milliseconds=rng.randint(50, 100)))
        
        # Add failure message to main keyword
        fail_msg_el = ET.SubElement(kw, "msg")
        fail_msg_el.set("timestamp", fail_ts)
        fail_msg_el.set("level", "FAIL")
        fail_msg_el.text = fail_msg
        
        # Add failing status to main keyword
        kw_status = ET.SubElement(kw, "status")
        kw_status.set("status", "FAIL")
        kw_status.set("starttime", start_s)
        kw_status.set("endtime", end_s)
        
        # Add inner keyword (the one that actually failed)
        inner_kw = ET.SubElement(kw, "kw")
        inner_kw.set("name", inner_kw_name)
        inner_kw.set("library", "SeleniumLibrary")
        
        inner_status = ET.SubElement(inner_kw, "status")
        inner_status.set("status", "FAIL")
        inner_status.set("starttime", inner_start)
        inner_status.set("endtime", end_s)
        
        # Add failing status to test
        outer_status = ET.SubElement(test_el, "status")
        outer_status.set("status", "FAIL")
        outer_status.set("starttime", start_s)
        outer_status.set("endtime", end_s)
    
    return test_el, status, end_dt


def build_run(n, config, rng):
    """
    Build complete XML and metadata for a single CI run.
    
    Creates:
      1. Robot Framework XML with all tests
      2. Metadata JSON with run statistics
    
    Process:
      1. Calculate target pass rate for this run
      2. Run dependency model to adjust fail probabilities
      3. Roll natural outcomes for each test
      4. Apply bidirectional correction to hit target pass rate
      5. Build XML for each test
      6. Generate run metadata
    
    Args:
        n: Run number (1-100)
        config: Configuration dictionary
        rng: Random number generator
    
    Returns:
        tuple: (root_element, metadata_dict)
            - root_element: ET.Element for <robot> root
            - metadata_dict: Build metadata for ci_metadata.json
    """
    # Calculate run timestamp
    run_dt = datetime.fromisoformat(config["start_date"]) + timedelta(hours=config["interval_hours"] * (n - 1))
    is_anomaly = n in config["anomaly_runs"]
    
    generated_s = fmt_ts(run_dt)
    suite_start = run_dt

    # Build XML root and suite
    
    root = ET.Element("robot")
    root.set("generator", "Robot 6.1.1 (Python 3.10.12)")
    root.set("generated", generated_s)
    root.set("rpa", "FALSE")
    root.set("schemaversion", "4")
    
    suite = ET.SubElement(root, "suite")
    suite.set("id", "s1")
    suite.set("name", config["suite_name"])
    suite.set("source", f"/opt/ci/tests/alpha/{config['suite_name']}.robot")
    
    # Suite setup keyword
    
    setup_end = suite_start + timedelta(milliseconds=110)
    kw_setup = ET.SubElement(suite, "kw")
    kw_setup.set("name", "Suite Setup")
    kw_setup.set("type", "setup")
    
    setup_msg = ET.SubElement(kw_setup, "msg")
    setup_msg.set("timestamp", generated_s)
    setup_msg.set("level", "INFO")
    setup_msg.text = f"Suite {config['suite_name']} initialized — team {config['team_name']}"
    
    setup_status = ET.SubElement(kw_setup, "status")
    setup_status.set("status", "PASS")
    setup_status.set("starttime", generated_s)
    setup_status.set("endtime", fmt_ts(setup_end))
    
    # Roll natural outcomes with dependency model
    
    cursor = setup_end + timedelta(milliseconds=200)
    passed = 0
    failed = 0
    
    target_pass_rate = run_pass_rate(n, config["num_runs"], config["anomaly_runs"], config["anomaly_pass_rate"])
    total_tests = len(TESTS)
    target_failures = round(total_tests * (1 - target_pass_rate))
    
    results = []
    natural_outcomes = {}  # name -> bool (True = pass)
    
    # Roll natural outcome for each test (with dependency adjustments)
    for test in TESTS:
        tid, name, feature_tag, priority_tag, category, fail_prob, *_ = test
        
        # Apply dependency risk model
        # If upstream tests failed, increase this test's fail probability
        dep_info = DEPENDENCIES.get(name)
        if dep_info:
            failed_dep_count = sum(
                1 for dep in dep_info["deps"]
                if natural_outcomes.get(dep) is False
            )
            if failed_dep_count > 0:
                # Multiplicative risk model:
                # effective = 1 - (1 - base) * (1 - weight * count)
                fail_prob = 1 - (1 - fail_prob) * (1 - dep_info["weight"] * failed_dep_count)
                fail_prob = min(fail_prob, 0.95)  # Cap at 95%
        
        # Roll outcome
        outcome = decide_outcome(category, fail_prob, is_anomaly, rng)
        natural_outcomes[name] = outcome
        results.append((test, outcome))
    
    natural_failures = sum(1 for _, outcome in results if not outcome)
    
    # Bidirectional correction to hit target pass rate
    
    extra_failures_needed = target_failures - natural_failures
    
    force_fail_indices = set()
    force_pass_indices = set()
    
    if extra_failures_needed > 0:
        # Too many passes — force some non-stable passing tests to fail
        candidates = [
            i for i, (test, outcome) in enumerate(results)
            if outcome and test[4] != "stable"
        ]
        n_force = min(extra_failures_needed, len(candidates))
        force_fail_indices = set(rng.sample(candidates, n_force))
    
    elif extra_failures_needed < 0:
        # Too many failures — force some failing tests to pass
        # Prefer flaky tests (more believable) over consistently_failing
        extra_passes_needed = -extra_failures_needed
        
        candidates_flaky = [
            i for i, (test, outcome) in enumerate(results)
            if not outcome and test[4] in ("flaky-mild", "flaky-moderate", "flaky-heavy")
        ]
        candidates_cf = [
            i for i, (test, outcome) in enumerate(results)
            if not outcome and test[4] == "consistently_failing"
        ]
        
        # Fill from flaky first, then consistently_failing if needed
        chosen = []
        for pool in (candidates_flaky, candidates_cf):
            still_needed = extra_passes_needed - len(chosen)
            if still_needed <= 0:
                break
            chosen += rng.sample(pool, min(still_needed, len(pool)))
        
        force_pass_indices = set(chosen)
    
    # Build XML for each test
    
    for i, (test, outcome) in enumerate(results):
        force_fail = i in force_fail_indices
        force_pass = i in force_pass_indices
        
        test_el, status, cursor = build_test_xml(
            test, outcome, n, config["num_runs"], is_anomaly, cursor, rng,
            force_fail=force_fail, force_pass=force_pass
        )
        
        suite.append(test_el)
        cursor += timedelta(milliseconds=rng.randint(100, 300))
        
        if status == "PASS":
            passed += 1
        else:
            failed += 1
    
    # Suite status
    
    suite_result = "FAIL" if failed > 0 else "PASS"
    suite_status_el = ET.SubElement(suite, "status")
    suite_status_el.set("status", suite_result)
    suite_status_el.set("starttime", generated_s)
    suite_status_el.set("endtime", fmt_ts(cursor))
    suite_status_el.set("passed", str(passed))
    suite_status_el.set("failed", str(failed))
    
    # Statistics
    
    stats_el = ET.SubElement(root, "statistics")
    
    total_el = ET.SubElement(stats_el, "total")
    stat_all = ET.SubElement(total_el, "stat")
    stat_all.set("pass", str(passed))
    stat_all.set("fail", str(failed))
    stat_all.text = "All Tests"
    
    tag_el = ET.SubElement(stats_el, "tag")
    stat_tag = ET.SubElement(tag_el, "stat")
    stat_tag.set("pass", str(passed))
    stat_tag.set("fail", str(failed))
    stat_tag.text = "alpha_regression"
    
    ET.SubElement(root, "errors")
    
    # Build metadata
    
    total = passed + failed
    meta = {
        "team": config["team_name"],
        "suite": config["suite_name"],
        "build_no": n,
        "timestamp": run_dt.isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate_pct": round(passed / total * 100, 1),
        "environment": "staging",
        "executor": f"jenkins-agent-{rng.randint(1, 9):02d}",
    }
    
    return root, meta


# MAIN GENERATION FUNCTION

def generate(config):
    """
    Generate all CI runs with progress output.
    
    Creates:
      - {num_runs} folders: TeamAlpha_build_001 to TeamAlpha_build_100
      - Each contains: output.xml and ci_metadata.json
    
    Progress is displayed every 10 runs to show generation is working.
    
    Args:
        config: Configuration dictionary with:
            - seed: Random seed for reproducibility
            - output_dir: Directory to write runs/
            - num_runs: Number of runs to generate (typically 100)
            - team_name: Team name for folder naming
    """
    rng = random.Random(config["seed"])
    out = config["output_dir"]
    num = config["num_runs"]
    
    os.makedirs(out, exist_ok=True)
    
    print(f"Generating {num} runs...")
    print(f"Output directory: {out}/")
    print(f"Random seed: {config['seed']}")
    print()
    
    for n in range(1, num + 1):
        # Create folder for this run
        folder = os.path.join(out, f"{config['team_name']}_build_{n:03d}")
        os.makedirs(folder, exist_ok=True)
        
        # Generate XML and metadata
        xml, meta = build_run(n, config, rng)
        
        # Write output.xml with pretty formatting
        _indent(xml)
        tree = ET.ElementTree(xml)
        with open(os.path.join(folder, "output.xml"), "w", encoding="utf-8") as f:
            tree.write(f, encoding="unicode", xml_declaration=False)
        
        # Write ci_metadata.json
        with open(os.path.join(folder, "ci_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        # Progress output every 10 runs
        if n % 10 == 0:
            print(f"  ✓ Run {n:3d}/{num}  "
                  f"pass={meta['passed']:2d}  "
                  f"fail={meta['failed']:2d}  "
                  f"pass_rate={meta['pass_rate_pct']:5.1f}%")
    
    print()
    print(f"✓ Done — {num} runs written to {out}/")
    print()
    print("Next steps:")
    print("  1. Run: python validate_output.py")
    print("  2. Verify all patterns are correct")
    print("  3. Proceed to Phase 2 (data pipeline)")


# COMMAND LINE INTERFACE

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Phase 1 Synthetic Test Data Generator",
        epilog="Example: python generate.py --num-runs 50 --seed 123"
    )
    p.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"],
                   help=f"Output directory (default: {DEFAULT_CONFIG['output_dir']})")
    p.add_argument("--num-runs", type=int, default=DEFAULT_CONFIG["num_runs"],
                   help=f"Number of runs to generate (default: {DEFAULT_CONFIG['num_runs']})")
    p.add_argument("--start-date", default=DEFAULT_CONFIG["start_date"],
                   help=f"Start date YYYY-MM-DD (default: {DEFAULT_CONFIG['start_date']})")
    p.add_argument("--interval", type=int, default=DEFAULT_CONFIG["interval_hours"],
                   dest="interval_hours",
                   help=f"Hours between runs (default: {DEFAULT_CONFIG['interval_hours']})")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                   help=f"Random seed (default: {DEFAULT_CONFIG['seed']})")
    p.add_argument("--team", default=DEFAULT_CONFIG["team_name"],
                   help=f"Team name (default: {DEFAULT_CONFIG['team_name']})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Merge command line args with defaults
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "output_dir": args.output_dir,
        "num_runs": args.num_runs,
        "start_date": args.start_date,
        "interval_hours": args.interval_hours,
        "seed": args.seed,
        "team_name": args.team,
    })
    
    # Run generation
    generate(cfg)