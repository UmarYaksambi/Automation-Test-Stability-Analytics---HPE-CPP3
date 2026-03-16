# Phase 1 – Synthetic Dataset Design Document

## Objective
This document defines the statistical signals embedded in the synthetic dataset to validate ML models in later phases.

## 1. Test Flakiness Distribution (DQ1)

The dataset intentionally contains a mix of stable, flaky, and consistently failing tests. 
This distribution allows the flakiness classifier in later phases to learn different 
stability behaviors rather than predicting a single dominant class.

| Test Name | Category | Fail Probability |
|:---|:---|:---|
| TC_Login_ValidCredentials | stable | 0% |
| TC_Login_InvalidPassword | stable | 0% |
| TC_Login_SessionTimeout | stable | 0% |
| TC_Login_AccountLockout | stable | 0% |
| TC_Login_MFAVerification | flaky-mild | 30% |
| TC_Dashboard_FilterByDate | stable | 0% |
| TC_Dashboard_Pagination | stable | 0% |
| TC_Dashboard_ExportChart | stable | 0% |
| TC_Dashboard_SearchBar | stable | 0% |
| TC_User_CreateAccount | stable | 0% |
| TC_User_EditProfile | stable | 0% |
| TC_User_DeleteAccount | stable | 0% |
| TC_User_PasswordReset | stable | 0% |
| TC_Login_SSORedirect | flaky-mild | 35% |
| TC_Dashboard_LoadWidget | flaky-moderate | 50% |
| TC_Dashboard_RefreshData | flaky-moderate | 55% |
| TC_User_BulkImport | flaky-heavy | 65% |
| TC_User_RoleAssignment | consistently_failing | 80% |
| TC_User_BatchExport | consistently_failing | 75% |
| TC_Login_OAuthCallback | consistently_failing | 70% |

## 2. Failure Category Balance (DQ2)

Failure messages are distributed across four categories (timeout, element, assertion, and data). 
Some tests contain both primary and secondary failure types to simulate realistic CI failures 
and allow clustering algorithms to identify distinct failure groups.

| Test Name | Primary Failure | Secondary Failure | Est. Failures |
|:---|:---|:---|:---|
| TC_Login_MFAVerification | timeout | assertion | 30 |
| TC_Login_SSORedirect | timeout | element | 35 |
| TC_Dashboard_LoadWidget | element | timeout | 50 |
| TC_Dashboard_RefreshData | assertion | data | 55 |
| TC_User_BulkImport | data | assertion | 65 |
| TC_User_RoleAssignment | assertion | data | 80 |
| TC_User_BatchExport | data | element | 75 |
| TC_Login_OAuthCallback | timeout | element | 70 |

**Total Estimated Failures:** ~460

## 3. Duration Patterns (DQ3)

Execution duration patterns are embedded in specific tests to simulate performance drift. 
These patterns are used to evaluate drift detection techniques in later phases.

| Test Name | Pattern |
|:---|:---|
| TC_Login_ValidCredentials | seasonal |
| TC_Login_InvalidPassword | normal |
| TC_Login_SessionTimeout | normal |
| TC_Login_AccountLockout | normal |
| TC_Login_MFAVerification | normal |
| TC_Dashboard_FilterByDate | normal |
| TC_Dashboard_Pagination | normal |
| TC_Dashboard_ExportChart | step_change |
| TC_Dashboard_SearchBar | normal |
| TC_User_CreateAccount | normal |
| TC_User_EditProfile | normal |
| TC_User_DeleteAccount | normal |
| TC_User_PasswordReset | normal |
| TC_Login_SSORedirect | normal |
| TC_Dashboard_LoadWidget | normal |
| TC_Dashboard_RefreshData | normal |
| TC_User_BulkImport | progressive |
| TC_User_RoleAssignment | normal |
| TC_User_BatchExport | normal |
| TC_Login_OAuthCallback | normal |

## 4. Programmed Anomalies
Runs **[36, 37]** simulate CI incidents with an expected pass rate of approximately 27%.
