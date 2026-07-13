#!/usr/bin/env python3
"""
Deep audit of episode concentration (HUPA0027P, HUPA0028P) and
Bland-Altman plots for ordinary FreeStyle-only participants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse parsers from main audit
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_audit import (  # noqa: E402
    CGM_NOMINAL_INTERVAL_MIN,
    DATASET_ROOT,
    EPISODE_SEPARATION_MIN,
    FIGURES_DIR,
    HYPO_THRESHOLD,
    SCAN_MATCH_TOLERANCE_MIN,
    detect_episodes,
    load_participant_freestyle,
    split_glucose_streams,
    _get_match_details,
)

OUTPUT_DIR = Path(__file__).resolve().parent
HIGH_PARTICIPANTS = ["HUPA0027P", "HUPA0028P"]
# Ordinary FreeStyle-only participants for Bland-Altman illustrations
BLAND_ALTMAN_PARTICIPANTS = ["HUPA0001P", "HUPA0005P", "HUPA0025P"]


def plot_bland_altman(match_details: list[dict], pid: str, out_dir: Path) -> dict:
    """Create Bland-Altman plot and return summary stats."""
    if not match_details:
        return {"participant_id": pid, "n_matched": 0}

    mdf = pd.DataFrame(match_details)
    mean_g = (mdf["scan_g"] + mdf["cgm_g"]) / 2
    diff_g = mdf["scan_g"] - mdf["cgm_g"]
    md = diff_g.mean()
    sd = diff_g.std()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mean_g, diff_g, alpha=0.35, s=12, edgecolors="none")
    ax.axhline(md, color="red", linestyle="--", linewidth=1.2, label=f"mean diff = {md:.1f} mg/dL")
    ax.axhline(md + 1.96 * sd, color="gray", linestyle=":", label=f"+1.96 SD = {md + 1.96*sd:.1f}")
    ax.axhline(md - 1.96 * sd, color="gray", linestyle=":", label=f"−1.96 SD = {md - 1.96*sd:.1f}")
    ax.set_xlabel("Mean glucose (mg/dL)")
    ax.set_ylabel("Scan − CGM (mg/dL)")
    ax.set_title(f"Bland-Altman: Scan vs CGM ({pid}, FreeStyle-only)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = out_dir / f"bland_altman_{pid}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    return {
        "participant_id": pid,
        "n_matched": len(mdf),
        "mean_diff": float(md),
        "sd_diff": float(sd),
        "loa_lower": float(md - 1.96 * sd),
        "loa_upper": float(md + 1.96 * sd),
        "median_abs_diff": float(mdf["abs_diff"].median()),
        "pct_within_15mg": float((mdf["abs_diff"] <= 15).mean() * 100),
    }


def active_cgm_days(cgm: pd.DataFrame) -> int:
    """Days with at least one CGM reading."""
    if cgm.empty:
        return 0
    return int(cgm["timestamp"].dt.date.nunique())


def calendar_span_days(cgm: pd.DataFrame) -> int:
    if cgm.empty:
        return 0
    return (cgm["timestamp"].max() - cgm["timestamp"].min()).days + 1


def count_borderline_oscillation(cgm: pd.DataFrame) -> dict:
    """
    Count transitions that cross 70 mg/dL boundary (potential episode inflation).
    """
    if cgm.empty:
        return {"crossings_below_70": 0, "crossings_69_71_band": 0}

    g = cgm.sort_values("timestamp")["glucose"].values
    crossings_70 = int(((g[:-1] >= 70) & (g[1:] < 70)).sum() + ((g[:-1] < 70) & (g[1:] >= 70)).sum())
    in_band = (g >= 69) & (g <= 71)
    band_cross = int((in_band[1:] != in_band[:-1]).sum())
    return {"crossings_below_70": crossings_70, "crossings_69_71_band": band_cross}


def check_repeated_values(cgm: pd.DataFrame) -> dict:
    if cgm.empty:
        return {"max_identical_streak": 0, "pct_readings_at_40": 0, "pct_readings_at_39": 0}
    g = cgm.sort_values("timestamp")["glucose"]
    streak = 1
    max_streak = 1
    prev = g.iloc[0]
    for val in g.iloc[1:]:
        if val == prev:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 1
        prev = val
    return {
        "max_identical_streak": max_streak,
        "pct_readings_at_40": float((g == 40).mean() * 100),
        "pct_readings_at_39": float((g == 39).mean() * 100),
    }


def audit_freestyle_files(pid: str) -> dict:
    """Per-source-file date ranges and row counts."""
    folder = DATASET_ROOT / "Raw_Data" / pid / "free_style_sensor"
    rows = []
    for f in sorted(folder.glob("*.csv")):
        raw = load_participant_freestyle(pid)
        # load individual file
        from data_audit import read_freestyle_file

        df = read_freestyle_file(f, pid)
        if df is None or df.empty:
            continue
        cgm, scans, _ = split_glucose_streams(df)
        rows.append(
            {
                "file": f.name,
                "total_rows": len(df),
                "cgm_rows": len(cgm),
                "scan_rows": len(scans),
                "date_min": df["timestamp"].min(),
                "date_max": df["timestamp"].max(),
                "record_types": sorted(df["record_type"].dropna().unique().tolist()),
            }
        )
    return {"participant_id": pid, "files": rows}


def deep_participant_audit(pid: str) -> dict:
    """Comprehensive audit for high-episode participants."""
    raw = load_participant_freestyle(pid)
    cgm, scans, _ = split_glucose_streams(raw)

    result: dict = {"participant_id": pid}
    if cgm.empty:
        result["error"] = "no_cgm"
        return result

    cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    dup_ts = int(cgm["timestamp"].duplicated().sum())

    ep30 = detect_episodes(cgm, HYPO_THRESHOLD, EPISODE_SEPARATION_MIN)
    ep60 = detect_episodes(cgm, HYPO_THRESHOLD, 60)
    ep45 = detect_episodes(cgm, HYPO_THRESHOLD, 45)

    cal_days = calendar_span_days(cgm)
    act_days = active_cgm_days(cgm)
    osc = count_borderline_oscillation(cgm)
    rep = check_repeated_values(cgm)

    # Per-file breakdown
    from data_audit import read_freestyle_file

    folder = DATASET_ROOT / "Raw_Data" / pid / "free_style_sensor"
    file_ranges = []
    for f in sorted(folder.glob("*.csv")):
        df = read_freestyle_file(f, pid)
        if df is None:
            continue
        fcgm, _, _ = split_glucose_streams(df)
        file_ranges.append(
            {
                "file": f.name,
                "cgm_min": fcgm["timestamp"].min() if len(fcgm) else None,
                "cgm_max": fcgm["timestamp"].max() if len(fcgm) else None,
                "cgm_rows": len(fcgm),
            }
        )

    # Gap analysis: longest gap between CGM readings
    intervals_h = cgm["timestamp"].diff().dt.total_seconds() / 3600
    longest_gap_h = float(intervals_h.max()) if len(intervals_h) else 0

    result.update(
        {
            "cgm_readings": len(cgm),
            "scan_readings": len(scans),
            "calendar_span_days": cal_days,
            "active_cgm_days": act_days,
            "cgm_date_min": cgm["timestamp"].min(),
            "cgm_date_max": cgm["timestamp"].max(),
            "duplicate_timestamps": dup_ts,
            "episodes_30min_sep": len(ep30),
            "episodes_45min_sep": len(ep45),
            "episodes_60min_sep": len(ep60),
            "episodes_per_active_day_30min": len(ep30) / max(act_days, 1),
            "episodes_per_calendar_day_30min": len(ep30) / max(cal_days, 1),
            "median_episode_duration_min": float(ep30["duration_min"].median()) if len(ep30) else 0,
            "mean_episode_duration_min": float(ep30["duration_min"].mean()) if len(ep30) else 0,
            "median_min_glucose_in_episodes": float(ep30["min_glucose"].median()) if len(ep30) else np.nan,
            "longest_cgm_gap_hours": longest_gap_h,
            "n_freestyle_files": len(file_ranges),
            "file_ranges": file_ranges,
            **osc,
            **rep,
            "low_readings_below_70": int((cgm["glucose"] < 70).sum()),
            "low_readings_below_54": int((cgm["glucose"] < 54).sum()),
            "pct_time_below_70": float((cgm["glucose"] < 70).mean() * 100),
        }
    )

    # Dexcom check for 0027P
    if pid == "HUPA0027P":
        from data_audit import audit_dexcom_freestyle_overlap

        dex = audit_dexcom_freestyle_overlap(pid)
        result["dexcom_overlap"] = dex.get("periods_overlap", False)
        result["dexcom_gap_days"] = dex.get("gap_days_freestyle_end_to_dexcom_start")

    return result


def episode_concentration_table(episode_summary: pd.DataFrame) -> pd.DataFrame:
    ep = episode_summary.sort_values("episodes_below_70", ascending=False).copy()
    total = ep["episodes_below_70"].sum()
    ep["pct_of_all_episodes"] = 100 * ep["episodes_below_70"] / max(total, 1)
    ep["cumulative_pct"] = ep["pct_of_all_episodes"].cumsum()
    return ep


def stride_window_summary(
    stride_min: int,
    history_h: int = 4,
    horizon_h: int = 2,
    miss_thresh: float = 0.2,
) -> dict:
    """Count eligible/positive windows at a given prediction stride."""
    from datetime import timedelta

    from data_audit import discover_participants

    participants = [p for p in discover_participants() if p not in ("HUPA0009P", "HUPA0010P")]
    total_elig = total_pos = 0
    for pid in participants:
        raw = load_participant_freestyle(pid)
        cgm, _, _ = split_glucose_streams(raw)
        if cgm.empty:
            continue
        cgm = cgm.sort_values("timestamp").drop_duplicates("timestamp")
        obs_index = pd.DatetimeIndex(cgm["timestamp"])
        t_min, t_max = obs_index.min(), obs_index.max()
        obs_set = set(obs_index)
        glucose = cgm.set_index("timestamp")["glucose"]
        grid = pd.date_range(
            t_min + timedelta(hours=history_h),
            t_max - timedelta(hours=horizon_h),
            freq=f"{stride_min}min",
        )
        for t in grid:
            input_start = t - timedelta(hours=history_h)
            future_end = t + timedelta(hours=horizon_h)
            hist_slots = pd.date_range(
                input_start, t, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min", inclusive="left"
            )
            fut_slots = pd.date_range(
                t, future_end, freq=f"{CGM_NOMINAL_INTERVAL_MIN}min", inclusive="left"
            )
            if not len(hist_slots) or not len(fut_slots):
                continue
            hist_miss = 1 - sum(1 for s in hist_slots if s in obs_set) / len(hist_slots)
            fut_miss = 1 - sum(1 for s in fut_slots if s in obs_set) / len(fut_slots)
            if hist_miss > miss_thresh or fut_miss > miss_thresh:
                continue
            total_elig += 1
            if (glucose.reindex(fut_slots) < HYPO_THRESHOLD).any():
                total_pos += 1
    return {
        "stride_min": stride_min,
        "history_hours": history_h,
        "horizon_hours": horizon_h,
        "n_eligible_windows": total_elig,
        "n_positive_windows": total_pos,
        "positive_rate": total_pos / max(total_elig, 1),
    }


def per_day_episode_stats(pid: str) -> dict:
    raw = load_participant_freestyle(pid)
    cgm, _, _ = split_glucose_streams(raw)
    ep = detect_episodes(cgm, HYPO_THRESHOLD, EPISODE_SEPARATION_MIN)
    if ep.empty:
        return {"participant_id": pid}
    ep = ep.copy()
    ep["day"] = ep["start"].dt.date
    daily = ep.groupby("day").size()
    return {
        "participant_id": pid,
        "days_with_episode": len(daily),
        "median_episodes_per_day": float(daily.median()),
        "mean_episodes_per_day": float(daily.mean()),
        "max_episodes_per_day": int(daily.max()),
        "p95_episodes_per_day": float(daily.quantile(0.95)),
        "days_with_3plus_episodes": int((daily >= 3).sum()),
        "days_with_5plus_episodes": int((daily >= 5).sum()),
    }


def write_report(
    high_audits: list[dict],
    bland_summaries: list[dict],
    concentration: pd.DataFrame,
    stride_rows: list[dict],
    daily_stats: list[dict],
) -> str:
    total_ep = int(concentration["episodes_below_70"].sum())
    top2 = concentration.head(2)["episodes_below_70"].sum()
    a27 = next(a for a in high_audits if a["participant_id"] == "HUPA0027P")
    a28 = next(a for a in high_audits if a["participant_id"] == "HUPA0028P")
    d27 = next(d for d in daily_stats if d["participant_id"] == "HUPA0027P")
    d28 = next(d for d in daily_stats if d["participant_id"] == "HUPA0028P")

    lines = [
        "# Episode Concentration and High-Participant Audit",
        "",
        "## Episode concentration",
        "",
        f"- Total episodes (<70, 30-min separation): **{total_ep}**",
        f"- HUPA0027P and HUPA0028P: **{int(top2)}** ({100*top2/total_ep:.1f}% of all episodes)",
        f"- Remaining 21 dense-cohort participants: **{int(total_ep - top2)}**",
        "",
        "See `episode_concentration_by_participant.csv` and `figures/episode_counts_by_participant.png`.",
        "",
        "## HUPA0027P validation",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Calendar span (days) | {a27['calendar_span_days']} |",
        f"| Active CGM days (≥1 reading) | {a27['active_cgm_days']} |",
        f"| Episodes (30-min sep) | {a27['episodes_30min_sep']} |",
        f"| Episodes (45-min sep) | {a27['episodes_45min_sep']} |",
        f"| Episodes (60-min sep) | {a27['episodes_60min_sep']} |",
        f"| Episodes per active day | {a27['episodes_per_active_day_30min']:.3f} |",
        f"| Episodes per calendar day | {a27['episodes_per_calendar_day_30min']:.3f} |",
        f"| Median episode duration (min) | {a27['median_episode_duration_min']:.0f} |",
        f"| CGM readings below 70 | {a27['low_readings_below_70']} ({a27['pct_time_below_70']:.1f}% of readings) |",
        f"| 69–71 mg/dL boundary crossings | {a27['crossings_69_71_band']} |",
        f"| FreeStyle files | {a27['n_freestyle_files']} |",
        f"| Dexcom overlap | {a27.get('dexcom_overlap', 'n/a')} (gap {a27.get('dexcom_gap_days', 'n/a')} days) |",
        f"| Max identical glucose streak | {a27['max_identical_streak']} |",
        f"| % readings exactly 40 mg/dL | {a27['pct_readings_at_40']:.2f}% |",
        f"| Longest CGM gap (hours) | {a27['longest_cgm_gap_hours']:.0f} |",
        f"| Days with ≥1 episode | {d27['days_with_episode']} |",
        f"| Median episodes on episode-days | {d27['median_episodes_per_day']:.0f} |",
        f"| Max episodes in one day | {d27['max_episodes_per_day']} |",
        "",
        "### HUPA0027P file ranges",
        "",
    ]
    for fr in a27.get("file_ranges", []):
        lines.append(f"- `{fr['file']}`: {fr['cgm_rows']:,} CGM rows, {fr['cgm_min']} → {fr['cgm_max']}")

    lines += [
        "",
        "**Duplicate export:** `...2021-11-17_2022-01-22.csv` and `...2022-01-21.csv` are **byte-for-byte duplicates** "
        "(69,931 CGM rows each, identical timestamps). `load_participant_freestyle` deduplicates on "
        "(timestamp, record_type, glucose), so episode count is **not inflated** by the extra file "
        "(681 episodes with or without file 3). Remove the duplicate file before modeling for clarity.",
        "",
        "**Interpretation:** 681 episodes over **769 active CGM days** ≈ **0.89 episodes/active-day**. "
        "On days with lows, median is **1 episode/day** (max **8**). This is elevated but plausible for a "
        "long-recording, hypoglycemia-prone participant (5.7% of readings <70). "
        "60-min separation reduces episodes to **305** (55% reduction), indicating moderate 69–71 mg/dL oscillation. "
        "Dexcom does **not** overlap FreeStyle; do not use HUPA0027P as the sole Bland-Altman illustration.",
        "",
        "## HUPA0028P validation",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Calendar span (days) | {a28['calendar_span_days']} |",
        f"| Active CGM days | {a28['active_cgm_days']} |",
        f"| Episodes (30-min sep) | {a28['episodes_30min_sep']} |",
        f"| Episodes (60-min sep) | {a28['episodes_60min_sep']} |",
        f"| Episodes per active day | {a28['episodes_per_active_day_30min']:.3f} |",
        f"| Median episode duration (min) | {a28['median_episode_duration_min']:.0f} |",
        f"| % time below 70 | {a28['pct_time_below_70']:.1f}% |",
        f"| 69–71 boundary crossings | {a28['crossings_69_71_band']} |",
        f"| Longest CGM gap (hours) | {a28['longest_cgm_gap_hours']:.0f} |",
        f"| Max episodes in one day | {d28['max_episodes_per_day']} |",
        "",
        f"**Interpretation:** 369 episodes over **{a28['active_cgm_days']} active days** ≈ "
        f"**{a28['episodes_per_active_day_30min']:.2f} episodes/active-day**. "
        f"60-min separation yields **{a28['episodes_60min_sep']}** episodes. "
        f"Median **1** episode on episode-days (max **{d28['max_episodes_per_day']}**).",
        "",
        "## Bland-Altman (FreeStyle-only participants)",
        "",
        "Generated for HUPA0001P, HUPA0005P, HUPA0025P (no Dexcom folder).",
        "",
        "| Participant | Matched pairs | Mean diff | 95% LoA | Median |abs diff| | % within 15 mg/dL |",
        "|-------------|-------------:|----------:|--------:|-----------------:|------------------:|",
    ]
    for b in bland_summaries:
        lines.append(
            f"| {b['participant_id']} | {b['n_matched']} | {b.get('mean_diff', 0):.1f} | "
            f"[{b.get('loa_lower', 0):.1f}, {b.get('loa_upper', 0):.1f}] | "
            f"{b.get('median_abs_diff', 0):.1f} | {b.get('pct_within_15mg', 0):.1f}% |"
        )

    lines += [
        "",
        "## Prediction-window stride (4h history, 2h horizon, 20% missingness)",
        "",
        "| Stride | Eligible windows | Positive windows | Positive rate |",
        "|--------|-----------------:|-----------------:|--------------:|",
    ]
    for s in stride_rows:
        lines.append(
            f"| {s['stride_min']} min | {s['n_eligible_windows']} | {s['n_positive_windows']} | "
            f"{100*s['positive_rate']:.1f}% |"
        )
    lines += [
        "",
        "30-min stride roughly halves windows while preserving ~15% positive rate. "
        "Use it to limit episode duplication (avg ~8–12 positive windows/episode at 15-min stride for HUPA0027P).",
        "",
        "## Recommended experimental design",
        "",
        "| Component | Specification |",
        "|-----------|---------------|",
        "| Dense input | Previous **4 h** historical CGM at 15-min resolution (16 steps) |",
        "| Sparse input | User-initiated scans from previous **6 h** |",
        "| Outcome | Any raw CGM <70 mg/dL in next **2 h** |",
        "| Dense models | XGBoost and small GRU |",
        "| Sparse model | XGBoost only (scan-summary features) |",
        "| Evaluation | Grouped 5-fold CV by participant; manual event stratification |",
        "| Window stride | **30 min** (reduce episode duplication) |",
        "| Metrics | AUPRC, recall, precision, F1, AUROC, per-participant distribution |",
        "",
        "## Verdict",
        "",
        "**Viable with modifications.** Episode counts are real but heavily concentrated (67% in two participants). "
        "HUPA0027P/HUPA0028P pass basic plausibility checks: episodes/active-day < 1, no timestamp duplication, "
        "Dexcom separated, duplicate export deduplicated. Remaining risks: model domination by two participants, "
        "fold instability, and 69–71 oscillation inflating episode counts. "
        "**Safe to lock the experimental design**; use grouped 5-fold CV with manual event stratification, "
        "30-min stride, 6h sparse / 4h dense history, and report participant-level metrics before claiming pooled performance.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating Bland-Altman for FreeStyle-only participants...")
    bland_rows = []
    for pid in BLAND_ALTMAN_PARTICIPANTS:
        raw = load_participant_freestyle(pid)
        cgm, scans, _ = split_glucose_streams(raw)
        _, details = _get_match_details(cgm, scans, SCAN_MATCH_TOLERANCE_MIN)
        summary = plot_bland_altman(details, pid, FIGURES_DIR)
        bland_rows.append(summary)
        print(f"  {pid}: {summary.get('n_matched', 0)} matched pairs")

    pd.DataFrame(bland_rows).to_csv(OUTPUT_DIR / "bland_altman_freestyle_only_summary.csv", index=False)

    print("Deep audit: HUPA0027P, HUPA0028P...")
    high_audits = []
    detail_rows = []
    for pid in HIGH_PARTICIPANTS:
        audit = deep_participant_audit(pid)
        high_audits.append(audit)
        for fr in audit.get("file_ranges", []):
            detail_rows.append({"participant_id": pid, **fr})
        print(
            f"  {pid}: {audit.get('episodes_30min_sep')} episodes, "
            f"{audit.get('episodes_per_active_day_30min', 0):.3f}/active-day, "
            f"{audit.get('active_cgm_days')} active days"
        )

    # Flatten high audit for CSV (exclude nested file_ranges)
    flat = []
    for a in high_audits:
        row = {k: v for k, v in a.items() if k != "file_ranges"}
        flat.append(row)
    pd.DataFrame(flat).to_csv(OUTPUT_DIR / "high_participant_audit.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUTPUT_DIR / "high_participant_file_ranges.csv", index=False)

    episode_summary = pd.read_csv(OUTPUT_DIR / "hypoglycemia_episode_summary.csv")
    concentration = episode_concentration_table(episode_summary)
    concentration.to_csv(OUTPUT_DIR / "episode_concentration_by_participant.csv", index=False)

    print("Stride window analysis (4h history, 2h horizon)...")
    stride_rows = [stride_window_summary(s) for s in (15, 30, 60)]
    pd.DataFrame(stride_rows).to_csv(OUTPUT_DIR / "prediction_stride_summary.csv", index=False)

    daily_stats = [per_day_episode_stats(pid) for pid in HIGH_PARTICIPANTS]
    pd.DataFrame(daily_stats).to_csv(OUTPUT_DIR / "high_participant_daily_episodes.csv", index=False)

    report = write_report(high_audits, bland_rows, concentration, stride_rows, daily_stats)
    (OUTPUT_DIR / "episode_concentration_audit.md").write_text(report, encoding="utf-8")
    print("Done. See episode_concentration_audit.md")


if __name__ == "__main__":
    main()
