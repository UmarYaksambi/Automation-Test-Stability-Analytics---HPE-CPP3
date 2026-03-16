import json
import random
from generate import TESTS, DEFAULT_CONFIG

def generate_design_doc():

    with open("design_doc.md", "w") as f:

        f.write("# Phase 1 – Synthetic Dataset Design Document\n\n")

        f.write("## Objective\n")
        f.write("This document defines the statistical signals embedded in the synthetic dataset ")
        f.write("to validate ML models in later phases.\n\n")

        # ----------------------
        # DQ1
        # ----------------------

        f.write("## 1. Test Flakiness Distribution (DQ1)\n\n")

        f.write("| Test Name | Category | Fail Probability |\n")
        f.write("|:---|:---|:---|\n")

        for t in TESTS:
            f.write(f"| {t[1]} | {t[4]} | {t[5]*100:.0f}% |\n")

        f.write("\n")

        # ----------------------
        # DQ2
        # ----------------------

        f.write("## 2. Failure Category Balance (DQ2)\n\n")

        f.write("| Test Name | Primary Failure | Secondary Failure | Est. Failures |\n")
        f.write("|:---|:---|:---|:---|\n")

        total_fails = 0

        for t in TESTS:
            if t[5] > 0:
                est = int(t[5] * DEFAULT_CONFIG["num_runs"])
                total_fails += est
                f.write(f"| {t[1]} | {t[7]} | {t[8]} | {est} |\n")

        f.write(f"\n**Total Estimated Failures:** ~{total_fails}\n\n")

        # ----------------------
        # DQ3
        # ----------------------

        f.write("## 3. Duration Patterns (DQ3)\n\n")

        f.write("| Test Name | Pattern |\n")
        f.write("|:---|:---|\n")

        for t in TESTS:
            name = t[1]
            pattern = t[6]
            f.write(f"| {name} | {pattern} |\n")

        # ----------------------
        # anomalies
        # ----------------------

        f.write("\n## 4. Programmed Anomalies\n")

        f.write(f"Runs **{DEFAULT_CONFIG['anomaly_runs']}** simulate CI incidents ")
        f.write(f"with an expected pass rate of approximately ")
        f.write(f"{DEFAULT_CONFIG['anomaly_pass_rate']*100:.0f}%.\n")

    print("design_doc.md generated successfully")


if __name__ == "__main__":
    generate_design_doc()