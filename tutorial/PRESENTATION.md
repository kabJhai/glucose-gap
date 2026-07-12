# Glucose Gap

### Presentation: What Is Lost Between Glucose Checks?

**Predicting near-term hypoglycemia from continuous and intermittent glucose observations**

**Format:** Copy each slide block into Google Slides or PowerPoint.  
**Speaker notes** appear under **Speaker notes:** below each slide.  
**Suggested length:** 18-20 minutes + 5 min Q&A

**Opening line:** My mother manages diabetes through food choices, insulin injections, and intermittent finger-prick testing. Watching that process made me wonder what happens in the hours between glucose checks, and whether machine learning can quantify the information lost during those gaps.

**Closing line:** Glucose Gap shows that the space between measurements is not merely missing data. It represents missing clinical visibility, and that loss can be measured.

**One-sentence thesis (use on title slide or closing):**

> This project shows that the information gap between glucose checks is measurable: models predict near-term hypoglycemia much better from continuous CGM history than from intermittent user-initiated scans, even when both come from the same sensor system.

## Slide 1: Personal motivation

**What happens between glucose checks?**

- Diabetes management is not just diagnosis; it is **continuous decision-making** between meals, insulin, and glucose checks
- Continuous sensors see glucose history all the time
- Intermittent checking only gives **snapshots**
- This tutorial turns that lived problem into a reproducible ML/DL analysis

[Your name] · [Course] · [Date]

**Speaker notes:**
- Open personally: watching a parent manage type 1 diabetes means living in the gaps between measurements, wondering what happened in the hour since the last check.
- That everyday anxiety became the research question: *what is lost when you only have snapshots instead of a continuous trace?*
- Frame the deck: you learned the method by building it; this presentation teaches a peer the story, the science, and how to rerun it.
- Close the slide with the one-sentence thesis if you have not put it on a title card.

## Slide 2: Clinical problem

- **Hypoglycemia** (glucose < 70 mg/dL) can become dangerous quickly
- A **2-hour warning** gives time to eat, adjust insulin, or seek help
- CGM makes proactive alerts possible, but not every moment is observed
- **Goal:** predict near-term hypoglycemia and measure what changes when history is intermittent

**Speaker notes:**
- 70 mg/dL is the standard research threshold used in this pipeline (`modeling/config.py`).
- Two hours is "actionable lead time": long enough to intervene, short enough to be relevant.
- Emphasize this is a tutorial/feasibility study on 22 participants, not a deployed clinical product.

## Slide 3: Dataset (HUPA-UCM)

| Element | Detail |
|---------|--------|
| Device | FreeStyle Libre 2 |
| Streams | Historical CGM slots + user-initiated scans |
| Participants (primary) | 22 with both CGM and scans |
| Key point | Scans are **not finger-pricks**; they are intermittent views of the **same sensor signal** |

Download: [Mendeley Data](https://data.mendeley.com/datasets/3hbcscwz44/1) · Full walkthrough: [`TUTORIAL.md`](TUTORIAL.md)

**Speaker notes:**
- HUPA-UCM provides ambulatory type 1 diabetes data: CGM, scans, insulin, meals, wearables.
- The critical conceptual point: we are **not** comparing CGM to capillary glucose. Scans and dense CGM are two sampling modes of the same FreeStyle signal.
- Fair comparison = same person, same clock time, same label; only the *input history* differs.
- Mention dataset license (CC BY 4.0) if presenting formally.

## Slide 4: Core question

**How much prediction performance is lost when continuous history becomes intermittent scan history?**

```
Continuous CGM history  ->  Dense model  --+
                                           +--> Same label: hypoglycemia in next 2 h
Intermittent scan history -> Sparse model -+
```

**Speaker notes:**
- This is the spine of the entire project. Every design choice serves this paired comparison.
- We are measuring an **information gap**, not asking whether sparse data is "useless."
- The answer quantifies how much near-term risk signal lives in the glucose trajectory between checks.

## Slide 5: Audit before modeling

**Never train on data you have not characterized.**

`feasibility_audit/data_audit.py` checks:

- Participant inventory and scan coverage
- Episode concentration (who drives positive labels?)
- Missingness and interpolation risk
- Leakage risks (features must use only past data)
- Window eligibility at 30-min stride

**Output:** `feasibility_audit/feasibility_report.md`

**Speaker notes:**
- Audit took ~8-9 minutes; it is Step 1 in `python run_tutorial.py`.
- Two participants (HUPA0027P, HUPA0028P) contribute ~67% of hypoglycemia episodes. We pre-register a sensitivity analysis before seeing model results.
- Teaching point for peers: always ask *who* is in your positive class before tuning hyperparameters.
- See audit figures in `feasibility_audit/figures/` after a full run.

## Slide 6: Paired design

**Same everything. Only the input changes.**

| Held constant | Varied |
|---------------|--------|
| 22 participants | Dense: 4 h continuous CGM |
| Identical prediction timestamps | Sparse: 6 h user-initiated scans |
| Identical hypoglycemia labels | |
| Identical 5-fold CV assignments | |

**Speaker notes:**
- This is what makes the comparison scientifically fair.
- `paired_windows.csv` is the single master table; dense and sparse features join onto the same rows without changing order or count.
- `fold_assignments.csv` is written once and reused by every model and sensitivity.

## Slide 7: Prediction setup

| Parameter | Value |
|-----------|-------|
| Dense input | Previous **4 h** CGM @ 15 min |
| Sparse input | Previous **6 h** scans |
| Label | Any glucose **< 70 mg/dL** in next **2 h** |
| Stride | **30 min** |
| Eligible windows | **1,260** (~**15%** positive) |

Timeline: `[4h dense input | 6h sparse input] -> prediction_time -> [2h label window]`

**Speaker notes:**
- Draw the timeline on a whiteboard: input window, prediction moment, horizon window.
- ≤20% missingness required in both input and label windows; this avoids training on empty sequences.
- 30-min stride limits overlap from a single hypoglycemia episode (vs 15-min stride).
- Canonical fields: `target_hypo_2h`, `has_prior_scan`, `scan_count_6h`.

## Slide 8: Leakage-safe validation

- **Grouped 5-fold CV** by participant; no person appears in both train and test
- **Event-aware** fold balancing; spread positive windows across folds
- Folds saved to `fold_assignments.csv` and **reused unchanged** across all models

**Speaker notes:**
- Autocorrelated glucose means random window splits inflate performance; grouped CV is non-negotiable for this task.
- Event-aware balancing helps, but cannot fully fix concentration when two participants dominate positives. Document that honestly.
- Inner validation (20% of training folds) tunes classification threshold only; no held-out peeking.
- Code: `modeling/cv_splits.py`.

## Slide 9: Features

**Dense model (continuous CGM):**  
Glucose summaries and trends: mean, median, min, slopes at 15/30/60/120 min, proportion below 70/80/90, missing-slot count, time since last valid reading, hour-of-day

**Sparse model (intermittent scans):**  
Scan count, last scan value, scan age, scan trend (last-two slope), missingness indicators

**Rule:** every feature uses observations **strictly before** `prediction_time`

**Speaker notes:**
- Leakage prevention is the most important engineering detail in healthcare time-series ML.
- Sparse windows with no prior scan exist; features are zero-filled and we stratify performance by `has_prior_scan` later.
- GRU (secondary experiment) receives a 16-step sequence with `[value, mask]` channels. See `modeling/features.py`.

## Slide 10: Models

| Experiment | Comparison | Why |
|------------|------------|-----|
| **1 (Primary)** | Dense XGBoost vs Sparse XGBoost | Continuous vs intermittent access |
| **2 (Secondary)** | Dense XGBoost vs small GRU | Tabular vs sequence on dense input |

**Primary metric: AUPRC** (positives are imbalanced, ~15%)  
Also report: AUROC, recall, precision, F1 (threshold tuned on inner validation)

**Speaker notes:**
- XGBoost with `scale_pos_weight` handles class imbalance; AUPRC is more informative than accuracy here.
- GRU is intentionally small (1 layer, 32 hidden units): tutorial comparison, not a benchmark chase.
- Underperformance of GRU vs XGBoost is a valid outcome on this sample size.
- Code: `modeling/train.py`.

## Slide 11: Results

| Model | AUPRC | AUROC |
|-------|------:|------:|
| **Dense XGBoost** | **0.659** | 0.847 |
| **Sparse XGBoost** | 0.127 | 0.406 |

**Dense minus sparse:** +0.532 AUPRC · **~81% relative loss** · Bootstrap 95% CI: [0.447, 0.649]

**The story is not "sparse data is useless."**  
**The story is: continuous access carries major predictive signal.**

**Speaker notes:**
- Lead with the interpretation, not just the table. Sparse scans retain only ~19% of dense AUPRC.
- Bootstrap CI excludes zero; the gap is robust at the participant level, not just window level.
- Sparse AUPRC > 0 means scans carry *some* signal, but far less than the continuous trajectory.
- Verify locally: `python scripts/verify_results.py`.

## Slide 12: Sensitivity analyses

**Remove the two dominant participants (HUPA0027P, HUPA0028P):**

| Model | AUPRC |
|-------|------:|
| Dense XGBoost | 0.704 |
| Sparse XGBoost | 0.205 |

**Sparse XGBoost by scan availability:**

| Stratum | AUPRC | n |
|---------|------:|--:|
| ≥1 prior scan in 6 h | 0.136 | 1,082 |
| No prior scan | 0.080 | 178 |

**Speaker notes:**
- Proves you checked whether results are participant-driven or scan-availability-driven.
- Excluding 27/28 raises both arms but preserves the large dense advantage; this is not a single-person artifact.
- No-scan stratum performs worst: without a recent scan, sparse features carry almost no signal.
- This is analytical maturity: show your instructor you stress-tested the headline number.

## Slide 13: Interpretation (SHAP)

![Dense XGBoost SHAP](../modeling_outputs/figures/shap_dense_xgb.png)

**Likely top drivers:**
- Recent glucose level
- Downward trend / slope
- Time spent near low range (proportion below thresholds)

**Speaker notes:**
- SHAP connects model output to clinical intuition: the model "pays attention" to patterns a careful caregiver would notice.
- Point to 2-3 features on the figure; exact ranking may vary slightly across runs.
- Sparse SHAP (`figures/shap_sparse_xgb.png`) often highlights scan recency, a different and weaker signal.
- Generate figures with `python -m modeling.train --skip-gru`.

## Slide 14: Replicability

**Anyone following the tutorial can rerun and verify.**

```bash
python run_tutorial.py              # full pipeline
python scripts/verify_results.py      # check against reference metrics
```

| Mechanism | File / location |
|-----------|-----------------|
| One-command orchestration | `run_tutorial.py` |
| Fixed hyperparameters | `modeling/config.py` (seed 42) |
| Saved folds (assigned once) | `fold_assignments.csv` |
| Intermediate artifacts | `paired_windows.csv`, feature CSVs, OOF predictions |
| Environment record | `run_manifest.json` |
| Peer checklist | [`TUTORIAL.md`](TUTORIAL.md) §6 |

**Speaker notes:**
- Assignment requirement: evaluate tutorial effectiveness for replicability.
- Strengths: scripted pipeline, pinned `requirements.txt`, verification script, saved folds.
- Limitations peers will hit: manual dataset download, macOS `libomp` for XGBoost, optional PyTorch for GRU.
- Re-running should reproduce identical `fold_assignments.csv` and metrics within ±0.02 AUPRC.

## Slide 15: Conclusion

**What this tutorial teaches:**

1. A **leakage-safe healthcare time-series pipeline**: audit first, paired design, grouped CV
2. A **measurable information gap**: intermittent monitoring makes hypoglycemia prediction substantially harder
3. A **reproducible artifact**: code, comments, verification targets, and this deck

**One-sentence thesis:**

> Models predict near-term hypoglycemia much better from continuous CGM history than from intermittent user-initiated scans, even when both come from the same sensor system.

**Speaker notes:**
- Return to the personal opening: the gap between checks is not just felt; it is quantifiable.
- Invite discussion: "Would you trust a scan-only alert app?" "What scan frequency would close the gap?"
- Leave repo link and dataset link on screen for Q&A.
- Thank your peer/instructor.

## Appendix: Slide-to-code mapping

| Slide | Primary code / doc |
|-------|-------------------|
| 3 | [`TUTORIAL.md`](TUTORIAL.md) §2 |
| 5 | `feasibility_audit/data_audit.py` |
| 6-7 | `modeling/windows.py` |
| 8 | `modeling/cv_splits.py` |
| 9 | `modeling/features.py` |
| 10-11 | `modeling/train.py` |
| 13 | `modeling_outputs/figures/shap_*.png` |
| 14 | [`TUTORIAL.md`](TUTORIAL.md) §6, `scripts/verify_results.py` |

## Appendix: Anticipated questions

| Question | Short answer |
|----------|--------------|
| Why AUPRC not accuracy? | ~15% positives; accuracy hides poor minority-class detection |
| Why not finger-stick comparison? | Scans are same-sensor snapshots, not capillary glucose |
| Is this clinically validated? | No; educational feasibility study; external validation needed |
| Can sparse ever match dense? | Not at observed scan frequency; more scans might narrow the gap |
