"""Generate human-readable modeling report for the tutorial."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modeling.config import OUTPUT_DIR


def _pick(metrics: pd.DataFrame, model: str) -> dict:
    rows = metrics[metrics["model"] == model]
    return rows.iloc[0].to_dict() if len(rows) else {}


def generate_modeling_report(out_dir: Path = OUTPUT_DIR) -> str:
    metrics_path = out_dir / "model_metrics.csv"
    if not metrics_path.exists():
        return "# Modeling results\n\nRun `python -m modeling.train` first.\n"

    metrics = pd.read_csv(metrics_path)
    manifest_path = out_dir / "run_manifest.json"
    summary_path = out_dir / "experiment_summary.json"
    paired_path = out_dir / "paired_comparison.csv"

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    paired = pd.read_csv(paired_path).iloc[0].to_dict() if paired_path.exists() else {}

    dense = _pick(metrics, "dense_xgb")
    sparse = _pick(metrics, "sparse_xgb")
    gru = _pick(metrics, "dense_gru")
    sens_d = _pick(metrics, "sens_dense_xgb")
    sens_s = _pick(metrics, "sens_sparse_xgb")
    has_scan = _pick(metrics, "sparse_has_scan")
    no_scan = _pick(metrics, "sparse_no_scan")

    n_windows = int(summary.get("n_windows", manifest.get("n_windows", dense.get("n_active", 0))))
    n_positive = int(summary.get("n_positive", manifest.get("n_positive", 0)))

    def metric_row(label: str, r: dict) -> str:
        if not r:
            return f"| {label} | n/a | n/a | n/a | n/a | n/a |"
        return (
            f"| {label} | {r.get('auprc', float('nan')):.3f} | "
            f"{r.get('auroc', float('nan')):.3f} | {r.get('recall', float('nan')):.3f} | "
            f"{r.get('precision', float('nan')):.3f} | {r.get('f1', float('nan')):.3f} |"
        )

    lines = [
        "# Modeling Results",
        "",
        "*Glucose Gap*: reproducible tutorial outputs.",
        "",
        "## Dataset windows (common 22-participant cohort)",
        "",
        f"- Eligible windows: **{n_windows}**",
        f"- Positive windows: **{n_positive}**",
        "- Same prediction timestamps for dense and sparse models",
        "- Grouped 5-fold CV with event-aware participant assignment",
        "",
        "## Experiment 1: Dense vs sparse XGBoost (primary)",
        "",
        "| Model | AUPRC | AUROC | Recall | Precision | F1 |",
        "|-------|------:|------:|-------:|----------:|---:|",
        metric_row("Dense XGBoost", dense),
        metric_row("Sparse XGBoost", sparse),
        "",
    ]

    if paired:
        lines += [
            f"**AUPRC advantage (dense − sparse): {paired.get('absolute_difference', float('nan')):+.3f}**",
            f"**Relative performance loss (sparse vs dense): {paired.get('relative_loss_pct', float('nan')):.1f}%**",
            f"**Participant-level bootstrap 95% CI on difference: "
            f"[{paired.get('boot_ci_lower', float('nan')):.3f}, {paired.get('boot_ci_upper', float('nan')):.3f}]**",
            "",
        ]

    lines += [
        "## Experiment 2: Dense XGBoost vs GRU",
        "",
        metric_row("Dense GRU", gru),
        "",
        "The GRU is intentionally small (1 layer, 16 steps). Underperformance vs XGBoost is a valid tutorial outcome.",
        "",
        "## Sensitivity analyses",
        "",
        "### Exclude HUPA0027P and HUPA0028P",
        "",
    ]
    if sens_d:
        lines.append(
            f"- Dense XGBoost AUPRC: **{sens_d.get('auprc', float('nan')):.3f}** "
            f"(n={int(sens_d.get('n_active', 0))} windows)"
        )
    if sens_s:
        lines.append(
            f"- Sparse XGBoost AUPRC: **{sens_s.get('auprc', float('nan')):.3f}** "
            f"(n={int(sens_s.get('n_active', 0))} windows)"
        )

    lines += ["", "### Sparse XGBoost by scan availability", ""]
    if has_scan:
        lines.append(
            f"- Windows with ≥1 prior scan (6 h): AUPRC **{has_scan.get('auprc', float('nan')):.3f}** "
            f"(n={int(has_scan.get('n_active', 0))})"
        )
    if no_scan:
        lines.append(
            f"- Windows with no prior scan: AUPRC **{no_scan.get('auprc', float('nan')):.3f}** "
            f"(n={int(no_scan.get('n_active', 0))})"
        )

    lines += [
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `paired_windows.csv` | Master paired prediction windows |",
        "| `fold_assignments.csv` | Event-aware CV folds (assigned once) |",
        "| `dense_features.csv` / `sparse_features.csv` | Tabular features per window |",
        "| `dense_sequences.npz` | 2-channel GRU sequences (value + mask) |",
        "| `oof_predictions.csv` | Out-of-fold probabilities (all models) |",
        "| `model_metrics.csv` | Pooled metrics for every experiment |",
        "| `paired_comparison.csv` | Dense vs sparse paired comparison + bootstrap CI |",
        "| `figures/shap_*.png` | SHAP summaries for XGBoost |",
        "| `run_manifest.json` | Seeds, config, package versions |",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python run_tutorial.py --model-only",
        "```",
        "",
    ]
    text = "\n".join(lines)
    (out_dir / "modeling_results.md").write_text(text, encoding="utf-8")
    return text
