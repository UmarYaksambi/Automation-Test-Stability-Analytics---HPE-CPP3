# Phase 1 Design Document

## Design Question 1 — Class Balance for the Flakiness Classifier

| Test Name | Category | Fail Probability |
|-----------|----------|-----------------|
| TC_Login_ValidCredentials | stable | 0% |
| TC_Login_InvalidPassword | stable | 0% |
| TC_Login_SessionTimeout | stable | 0% |
| TC_Login_AccountLockout | stable | 0% |
| TC_Dashboard_FilterByDate | stable | 0% |
| TC_Dashboard_Pagination | stable | 0% |
| TC_Dashboard_ExportChart | stable | 0% |
| TC_Dashboard_SearchBar | stable | 0% |
| TC_User_CreateAccount | stable | 0% |
| TC_User_EditProfile | stable | 0% |
| TC_User_DeleteAccount | stable | 0% |
| TC_User_PasswordReset | stable | 0% |
| TC_Login_SSORedirect | flaky-mild | 35% |
| TC_Login_MFAVerification | flaky-mild | 30% |
| TC_Dashboard_LoadWidget | flaky-moderate | 50% |
| TC_Dashboard_RefreshData | flaky-moderate | 55% |
| TC_User_BulkImport | flaky-heavy | 65% |
| TC_User_RoleAssignment | consistently failing | 80% |
| TC_User_BatchExport | consistently failing | 75% |
| TC_Login_OAuthCallback | consistently failing | 70% |

**Category summary:** 12 stable · 2 flaky-mild · 2 flaky-moderate · 1 flaky-heavy · 3 consistently failing

**Rationale:** Five distinct fail probabilities (0.30, 0.35, 0.50, 0.55, 0.65) give the flakiness classifier a meaningful spectrum to learn from. The minimum spec requirement is two levels (mild ~30%, heavy ~60%); five levels gives the Random Forest classifier a richer feature space and produces a more informative feature importance chart in Phase 4. The three consistently-failing tests (0.70–0.80) use different probabilities so their individual fail-rate features remain distinguishable to the classifier.

**Pass rate curve interaction:** Flaky tests (SSORedirect, MFAVerification, LoadWidget, RefreshData, BulkImport) always draw from their own independent probability, the suite-level pass rate curve does NOT suppress or amplify their individual rolls. Consistently-failing tests (RoleAssignment, BatchExport, OAuthCallback) are governed by the suite-level curve: at anomaly runs 36–37 (target suite pass rate ~25%), their already-high individual fail probabilities are further scaled upward so the anomaly appears clearly in per-test data as well as in the aggregate. Stable tests have 0% individual fail probability and are unaffected by the curve except that their PASS results contribute to the suite-level count.

**ML impact (Phase 4 — ML 1 Flakiness Classifier):** The five-level spectrum means `fail_rate_last_10` and `status_changes` will produce a continuous gradient of values rather than three clumped groups. Feature importance output will clearly rank recent-window rate above all-time rate, validating the insight that short-term behaviour is more predictive than long-term averages.

---

## Design Question 2 — Category Balance for Failure Clustering

Expected failures per test over 100 runs, and failure type distribution:

| Test Name | Primary Failure Type | Secondary Failure Type | Est. failures in 100 runs |
|-----------|---------------------|----------------------|--------------------------|
| TC_Login_SSORedirect | timeout (70%) | element (30%) | ~35 |
| TC_Login_MFAVerification | timeout (70%) | assertion (30%) | ~30 |
| TC_Dashboard_LoadWidget | element (80%) | timeout (20%) | ~50 |
| TC_Dashboard_RefreshData | assertion (60%) | data (40%) | ~55 |
| TC_User_BulkImport | data (70%) | assertion (30%) | ~65 |
| TC_User_RoleAssignment | assertion (65%) | data (35%) | ~80 |
| TC_User_BatchExport | data (65%) | element (35%) | ~75 |
| TC_Login_OAuthCallback | timeout (70%) | element (30%) | ~70 |

**Estimated category totals across ~460 total failures:**

| Category | Calculation | Estimated count | % |
|----------|-------------|-----------------|---|
| timeout  | 35×0.70 + 30×0.70 + 50×0.20 + 70×0.70 | ~105 | ~23% |
| element  | 35×0.30 + 50×0.80 + 75×0.35 + 70×0.30 | ~98  | ~21% |
| assertion| 30×0.30 + 55×0.60 + 65×0.30 + 80×0.65 | ~114 | ~25% |
| data     | 55×0.40 + 65×0.70 + 75×0.65 + 80×0.35 | ~144 | ~31% |
| **Total**| | **~461** | **100%** |

No single category exceeds 40%. The distribution is balanced enough for K-Means (k=4) to form four genuine clusters rather than one dominant cluster and three sparse ones. The `data` category is slightly higher (~31%) because the three highest-volume consistently-failing tests (BulkImport 65/run, BatchExport 75/run, RoleAssignment 80/run) all carry data as a primary or secondary type. This is intentional and within the spec's ≤40% ceiling.

**ML impact (Phase 4 — ML 2 Failure Clustering):** The 21–31% spread across four categories means TF-IDF + K-Means will converge on four genuine clusters. If one category held 60%+ the dominant cluster would absorb boundary messages from adjacent categories, making the pie chart misleading. The mixed primary/secondary split per test also ensures every cluster contains messages from multiple tests, preventing the degenerate case where a cluster is entirely sourced from one test name.

---

## Design Question 3 — Duration Patterns for Drift Detection

| Test Name | Duration Pattern | Normal range (s) | Degraded range (s) |
|-----------|-----------------|-----------------|-------------------|
| TC_Login_ValidCredentials | seasonal (period = 2) | 2.0–3.5 (even runs) | 4.5–6.5 (odd runs) |
| TC_Dashboard_ExportChart | step change at run 51 | 3.0–5.0 (runs 1–50) | 12.0–15.0 (runs 51–100) |
| TC_User_BulkImport | progressive drift | 10.0–14.0 (runs 1–40) · 18.0–24.0 (runs 41–65) | 28.0–36.0 (runs 66–100) |
| all other 17 tests | normal (baseline) | 1.2–8.5 | base + 5–15 on FAIL |

**Why three distinct patterns are required for ML Phase 4 (ML 4 — Duration Drift):**

- **Progressive drift** (BulkImport): Duration increases gradually across three phases. A static threshold set at run 1 never fires because the change is incremental. The correct detector is a rolling Z-score with a baseline window anchored to early runs (1–20); by runs 80–100 the test's duration is 2–3× the baseline mean, which exceeds any reasonable Z-score cutoff. This is the leading-indicator use case: the test slows for 30 runs before it starts failing consistently.

- **Step change** (ExportChart): Duration is normal for runs 1–50, then jumps 3–4× at run 51 with no gradual transition. A rolling Z-score with a short window (e.g., 10 runs) fires immediately at run 51 because the delta from the trailing mean is enormous. This is the simplest and most satisfying anomaly to detect, it demonstrates that the model catches what a human would notice glancing at a chart.

- **Seasonal** (ValidCredentials): Even runs are fast (2.0–3.5s), odd runs are slow (4.5–6.5s), with no underlying degradation. A rolling Z-score will flag every odd run as an anomaly, a false positive rate of 50%. The correct approach is autocorrelation (period=2) or conditioning on run parity (even/odd). This is the key ML insight: the algorithm that correctly handles step changes and progressive drift produces only noise on seasonal data, demonstrating that one-size-fits-all detectors fail in practice.

**Anomaly run interaction (runs 36–37):** The per-test duration patterns above are independent of the suite-level anomaly spike. Failing tests on runs 36–37 will naturally have elevated durations (base + 5–15s on FAIL) but this does not introduce an extra duration pattern, it is already captured by the "baseline + failure overhead" rule for normal tests.

**ML impact (Phase 4 — ML 4 Duration Drift):** The three patterns guarantee that Phase 4 produces three qualitatively different outputs: a true positive on ExportChart (step), a delayed true positive on BulkImport (drift), and a false-positive discussion on ValidCredentials (seasonal). Together these give the report a real analytical conclusion, not just "it works" but "it works for two patterns and fails for one, and here is why."