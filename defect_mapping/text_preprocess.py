"""
Text preprocessing utilities for defect-to-test-run mapping.
"""

import re
from typing import Optional


_BOILERPLATE_HEADERS = [
    r"Reproduction\s*\(Generic\):.*?(?=\n\n|\nExpected|\nActual|\nImpact|\nEnvironment|\nRequested|\Z)",
    r"Environment\s*\(Synthetic\):.*?(?=\n\n|\nRequested|\Z)",
    r"Requested Action:.*?(?=\n\n|\Z)",
]


def clean_defect_text(defect: dict) -> str:
    summary = defect.get("summary", "")
    description = defect.get("description", "")

    # Remove boilerplate sections from description
    cleaned_desc = description
    for pattern in _BOILERPLATE_HEADERS:
        cleaned_desc = re.sub(pattern, "", cleaned_desc, flags=re.DOTALL | re.IGNORECASE)

    # Combine summary and cleaned description
    combined = f"{summary}\n{cleaned_desc}"

    # Normalize whitespace
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    combined = re.sub(r"[ \t]+", " ", combined)
    combined = combined.strip()

    return combined


def clean_failure_text(failure_msg: Optional[str], failure_kw: Optional[str] = None) -> str:
    parts = []

    if failure_kw:
        parts.append(f"Keyword: {failure_kw}")

    if failure_msg:
        # Normalize common separators
        msg = failure_msg.replace("—", "-").replace("–", "-")
        # Remove excessive whitespace
        msg = re.sub(r"\s+", " ", msg).strip()
        parts.append(msg)

    return " | ".join(parts) if parts else ""


def extract_tc_names(text: str) -> list[str]:
    matches = re.findall(r"\bTC_[A-Za-z0-9_]+\b", text)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


def extract_failure_keyword(description: str) -> Optional[str]:
    """
    Extract the failure keyword from a JIRA defect description.

    Looks for lines like:
      - Keyword: Verify Row Count
      - Keywords: Wait Until Element Is Visible / Wait Until Redirect Completes
    """
    # Single keyword
    match = re.search(r"-\s*Keyword:\s*(.+?)(?:\n|$)", description)
    if match:
        return match.group(1).strip()

    # Multiple keywords (consolidated defects)
    match = re.search(r"-\s*Keywords?:\s*(.+?)(?:\n|$)", description)
    if match:
        return match.group(1).strip()

    return None


def extract_sample_message(description: str) -> Optional[str]:
    """
    Extract the 'Sample Message' line(s) from a JIRA defect description.

    Handles both single-message and multi-message formats:
      - Sample Message: CSV import contained 2 rows...
      - Sample Messages:
          * MFA overlay still visible after 45s
          * OAuth redirect not completed after 30s
    """
    # Multi-message format (bulleted list)
    multi_match = re.search(
        r"-\s*Sample Messages?:\s*\n((?:\s*\*\s*.+\n?)+)",
        description,
        re.IGNORECASE,
    )
    if multi_match:
        lines = multi_match.group(1).strip().split("\n")
        messages = []
        for line in lines:
            cleaned = re.sub(r"^\s*\*\s*", "", line).strip()
            if cleaned:
                messages.append(cleaned)
        return " | ".join(messages) if messages else None

    # Single message format
    single_match = re.search(
        r"-\s*Sample Message:\s*(.+?)(?:\n|$)", description, re.IGNORECASE
    )
    if single_match:
        return single_match.group(1).strip()

    return None


# ─── Failure Category Classification ────────────────────────────────
# Uses the same keyword rules as the existing dashboard for consistency.

_CATEGORY_RULES = [
    ("timeout",     [
        "still visible after", "timeout", "timed out", "did not respond",
        "did not complete within", "exceeded", "waited",
    ]),
    ("element",     [
        "not found", "element", "locator", "no match", "not visible",
        "missing", "not found in dom",
    ]),
    ("assertion",   [
        "expected http", "assertion", "should be equal", "mismatch",
        "expected field", "schema", "not present in response",
    ]),
    ("data",        [
        "csv export", "csv import", "row count", "rows", "record",
        "0 bytes", "empty file", "data inconsistency", "partial data",
        "row-count", "processed 2",
    ]),
    ("environment", [
        "environment", "unreachable", "connection refused",
        "infrastructure", "connection pool", "health check",
    ]),
]


def classify_failure_category(text: str) -> str:
    """
    Classify text into one of 5 failure categories using keyword matching.

    Categories: timeout, element, assertion, data, environment.
    Falls back to 'unknown' if no keywords match.
    """
    if not text:
        return "unknown"

    lower = text.lower()

    # Score each category by number of matching keywords
    best_category = "unknown"
    best_score = 0

    for category, keywords in _CATEGORY_RULES:
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_category = category

    return best_category
