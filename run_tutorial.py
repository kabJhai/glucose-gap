#!/usr/bin/env python3
"""
Reproducible end-to-end tutorial pipeline (course assignment deliverable).

Teaches: feasibility audit → leakage-safe paired modeling → replicability manifest.

Full walkthrough with speaker notes:
  tutorial/TUTORIAL.md
  tutorial/PRESENTATION.md

Usage (from project root):
  python run_tutorial.py              # full pipeline
  python run_tutorial.py --audit-only
  python run_tutorial.py --model-only
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _python() -> str:
    venv = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def write_run_manifest(steps: list[str]) -> None:
    import json

    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import xgboost

        versions = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        }
    except ImportError:
        versions = {}

    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        pass

    manifest = {
        "project": "Glucose Gap: What Is Lost Between Glucose Checks?",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "steps_completed": steps,
        "package_versions": versions,
        "commands": {
            "audit": f"{_python()} feasibility_audit/data_audit.py",
            "modeling": f"{_python()} -m modeling.train",
        },
    }
    out = PROJECT_ROOT / "modeling_outputs" / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible HUPA hypoglycemia tutorial")
    parser.add_argument("--audit-only", action="store_true", help="Run feasibility audit only")
    parser.add_argument("--model-only", action="store_true", help="Run modeling only")
    args = parser.parse_args()

    py = _python()
    steps: list[str] = []

    if not args.model_only:
        run_step("Step 1: Feasibility audit", [py, "feasibility_audit/data_audit.py"])
        steps.append("feasibility_audit")

    if not args.audit_only:
        run_step("Step 2: Modeling experiments", [py, "-m", "modeling.train"])
        steps.append("modeling")

    write_run_manifest(steps)

    if not args.audit_only:
        from modeling.report import generate_modeling_report

        report = generate_modeling_report()
        print("\n=== Step 3: Modeling report ===")
        print(report.splitlines()[0])

    print("\nTutorial pipeline complete.")
    print("Assignment docs: tutorial/TUTORIAL.md · tutorial/PRESENTATION.md")
    if not args.audit_only:
        print("Verify metrics:  python scripts/verify_results.py")


if __name__ == "__main__":
    main()
