# Episode Concentration and High-Participant Audit

## Episode concentration

- Total episodes (<70, 30-min separation): **1575**
- HUPA0027P and HUPA0028P: **1050** (66.7% of all episodes)
- Remaining 21 dense-cohort participants: **525**

See `episode_concentration_by_participant.csv` and `figures/episode_counts_by_participant.png`.

## HUPA0027P validation

| Metric | Value |
|--------|------:|
| Calendar span (days) | 787 |
| Active CGM days (≥1 reading) | 769 |
| Episodes (30-min sep) | 681 |
| Episodes (45-min sep) | 444 |
| Episodes (60-min sep) | 305 |
| Episodes per active day | 0.886 |
| Episodes per calendar day | 0.865 |
| Median episode duration (min) | 30 |
| CGM readings below 70 | 3975 (5.7% of readings) |
| 69–71 mg/dL boundary crossings | 1440 |
| FreeStyle files | 3 |
| Dexcom overlap | False (gap 276 days) |
| Max identical glucose streak | 23 |
| % readings exactly 40 mg/dL | 0.12% |
| Longest CGM gap (hours) | 190 |
| Days with ≥1 episode | 427 |
| Median episodes on episode-days | 1 |
| Max episodes in one day | 8 |

### HUPA0027P file ranges

- `HUPA0027P_free_style_sensor_2020-06-26_2020-11-04.csv`: 11,445 CGM rows, 2020-06-26 13:20:00 → 2020-11-04 23:40:00
- `HUPA0027P_free_style_sensor_2021-11-17_2022-01-22.csv`: 69,931 CGM rows, 2019-11-27 21:40:00 → 2022-01-22 14:33:00
- `HUPA0027P_free_style_sensor_2022-01-21.csv`: 69,931 CGM rows, 2019-11-27 21:40:00 → 2022-01-22 14:33:00

**Duplicate export:** `...2021-11-17_2022-01-22.csv` and `...2022-01-21.csv` are **byte-for-byte duplicates** (69,931 CGM rows each, identical timestamps). `load_participant_freestyle` deduplicates on (timestamp, record_type, glucose), so episode count is **not inflated** by the extra file (681 episodes with or without file 3). Remove the duplicate file before modeling for clarity.

**Interpretation:** 681 episodes over **769 active CGM days** ≈ **0.89 episodes/active-day**. On days with lows, median is **1 episode/day** (max **8**). This is elevated but plausible for a long-recording, hypoglycemia-prone participant (5.7% of readings <70). 60-min separation reduces episodes to **305** (55% reduction), indicating moderate 69–71 mg/dL oscillation. Dexcom does **not** overlap FreeStyle; do not use HUPA0027P as the sole Bland-Altman illustration.

## HUPA0028P validation

| Metric | Value |
|--------|------:|
| Calendar span (days) | 593 |
| Active CGM days | 580 |
| Episodes (30-min sep) | 369 |
| Episodes (60-min sep) | 172 |
| Episodes per active day | 0.636 |
| Median episode duration (min) | 30 |
| % time below 70 | 4.4% |
| 69–71 boundary crossings | 828 |
| Longest CGM gap (hours) | 341 |
| Max episodes in one day | 6 |

**Interpretation:** 369 episodes over **580 active days** ≈ **0.64 episodes/active-day**. 60-min separation yields **172** episodes. Median **1** episode on episode-days (max **6**).

## Bland-Altman (FreeStyle-only participants)

Generated for HUPA0001P, HUPA0005P, HUPA0025P (no Dexcom folder).

| Participant | Matched pairs | Mean diff | 95% LoA | Median |abs diff| | % within 15 mg/dL |
|-------------|-------------:|----------:|--------:|-----------------:|------------------:|
| HUPA0001P | 215 | -1.2 | [-23.6, 21.2] | 6.0 | 86.5% |
| HUPA0005P | 220 | 1.3 | [-15.6, 18.3] | 4.0 | 91.4% |
| HUPA0025P | 308 | -0.3 | [-12.6, 12.0] | 3.0 | 96.1% |

## Prediction-window stride (4h history, 2h horizon, 20% missingness)

| Stride | Eligible windows | Positive windows | Positive rate |
|--------|-----------------:|-----------------:|--------------:|
| 15 min | 2510 | 372 | 14.8% |
| 30 min | 1260 | 190 | 15.1% |
| 60 min | 632 | 96 | 15.2% |

30-min stride roughly halves windows while preserving ~15% positive rate. Use it to limit episode duplication (avg ~8–12 positive windows/episode at 15-min stride for HUPA0027P).

## Recommended experimental design

| Component | Specification |
|-----------|---------------|
| Dense input | Previous **4 h** historical CGM at 15-min resolution (16 steps) |
| Sparse input | User-initiated scans from previous **6 h** |
| Outcome | Any raw CGM <70 mg/dL in next **2 h** |
| Dense models | XGBoost and small GRU |
| Sparse model | XGBoost only (scan-summary features) |
| Evaluation | Grouped 5-fold CV by participant; manual event stratification |
| Window stride | **30 min** (reduce episode duplication) |
| Metrics | AUPRC, recall, precision, F1, AUROC, per-participant distribution |

## Verdict

**Viable with modifications.** Episode counts are real but heavily concentrated (67% in two participants). HUPA0027P/HUPA0028P pass basic plausibility checks: episodes/active-day < 1, no timestamp duplication, Dexcom separated, duplicate export deduplicated. Remaining risks: model domination by two participants, fold instability, and 69–71 oscillation inflating episode counts. **Safe to lock the experimental design**; use grouped 5-fold CV with manual event stratification, 30-min stride, 6h sparse / 4h dense history, and report participant-level metrics before claiming pooled performance.
