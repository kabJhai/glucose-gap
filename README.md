# Glucose Gap: What Is Lost Between Glucose Checks?

### Predicting Hypoglycemia from Continuous and Intermittent Glucose Observations

A reproducible healthcare ML/DL study measuring how much near-term hypoglycemia prediction performance is lost when a model sees only intermittent user-initiated glucose scans instead of continuous CGM history.

This project uses the [HUPA-UCM Diabetes Dataset](https://data.mendeley.com/datasets/3hbcscwz44/1) and compares leakage-safe XGBoost and GRU models under continuous and intermittent glucose-observation conditions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-orange)
![DL](https://img.shields.io/badge/DL-GRU-red)
![Explainability](https://img.shields.io/badge/Explainability-SHAP-purple)
![Dataset](https://img.shields.io/badge/Dataset-HUPA--UCM-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Repository:** https://github.com/kabJhai/glucose-gap

Measuring the predictive cost of intermittent glucose monitoring with leakage-safe machine learning and deep learning.

## Headline result

Using paired prediction windows from the same participants, timestamps, labels, and cross-validation folds:

| Model | Input condition | AUPRC | AUROC |
|---|---|---:|---:|
| XGBoost | Continuous CGM history | 0.659 | 0.847 |
| XGBoost | Intermittent scans | 0.127 | 0.406 |

The intermittent model experienced an absolute AUPRC reduction of **0.532**. Sparse AUPRC (**0.127**) fell below the ~15% prevalence baseline and below a simple latest-scan risk rule (~0.19), indicating that occasional snapshots did not support reliable two-hour prediction in this setup.

On the continuous side, a one-feature **latest-CGM baseline** (AUPRC **0.672**) slightly edges dense XGBoost (**0.659**): continuous access carries the predictive signal, and most of it is already in the current reading — engineered summaries do not add much on this small cohort.

These are reference results using seed 42 (`python -m modeling.train --skip-gru`). Small differences may occur across software environments. Confirm with `python scripts/verify_results.py`.

## Why this project exists

This project was inspired by watching my mother manage diabetes through food choices, insulin self-injections, and intermittent finger-prick glucose testing.

Her experience raised a practical question:

> What predictive information is lost between glucose checks?

Continuous glucose monitoring provides a dense history of glucose behavior. Intermittent checking provides only snapshots. This project measures how that difference in data availability affects a machine-learning model's ability to anticipate hypoglycemia.

The HUPA-UCM dataset uses FreeStyle Libre historical CGM readings and user-initiated sensor scans. The scans are not finger-prick measurements, so the experiment should be interpreted as a comparison between continuous and intermittent access to the same sensor signal.

## Research question

How much does near-term hypoglycemia-prediction performance decline when a model receives only intermittent user-initiated glucose scans instead of continuous historical CGM?

## Key findings

1. Continuous glucose history carries substantially more predictive information than intermittent scan history.
2. Sparse XGBoost performed near or below naive baselines (prevalence ~0.15; latest-scan rule ~0.19), not just below the dense model.
3. Sparse prediction is limited not only by scan frequency, but by whether a recent scan exists at prediction time.
4. Raw data auditing is essential. Using the preprocessed glucose table would have introduced interpolation-related label distortion.
5. Patient-level grouped cross-validation is necessary to avoid optimistic performance estimates.
6. A more complex model is not automatically better: sparse XGBoost underperformed a one-feature latest-scan baseline.

## Pipeline

```text
Raw HUPA-UCM files
        |
Participant and data-quality audit
        |
Raw CGM and scan extraction
        |
Leakage-safe paired prediction windows
        |
Event-aware grouped participant folds
        |
Dense and sparse feature engineering
        |
XGBoost and GRU training
        |
Out-of-fold evaluation
        |
Sensitivity analysis and SHAP interpretation
        |
Reproducibility report
```

## Experimental design

| Parameter | Locked design |
|---|---|
| Dataset | HUPA-UCM Diabetes Dataset |
| Primary cohort | 22 participants with historical CGM and user scans |
| Dense input | Previous 4 hours of CGM at 15-minute resolution |
| Sparse input | User-initiated scans from the previous 6 hours |
| Prediction horizon | Next 2 hours |
| Prediction stride | 30 minutes |
| Outcome | Any raw historical CGM value below 70 mg/dL |
| Missingness rule | No more than 20% missing CGM slots |
| Validation | Grouped 5-fold cross-validation by participant |
| Primary metric | AUPRC |
| Secondary metrics | Recall, precision, F1, AUROC, specificity |
| Interpretation | SHAP for XGBoost |
| Random seed | 42 |

## Experiments

### Experiment 1: Continuous versus intermittent observation

A paired comparison using the same participants, prediction timestamps, labels, and cross-validation folds.

**Models:**
- Dense XGBoost using continuous CGM-derived features
- Sparse XGBoost using intermittent scan-derived features

This is the primary experiment.

### Experiment 2: Tabular ML versus sequence DL

**Models:**
- Dense XGBoost
- Small one-layer GRU

Both use the same dense CGM prediction windows. The GRU experiment tests whether direct sequence modeling adds value beyond engineered temporal features.

## Dataset

This project uses the [HUPA-UCM Diabetes Dataset](https://data.mendeley.com/datasets/3hbcscwz44/1), released under CC BY 4.0.

The dataset includes data from 25 people with Type 1 diabetes and contains:
- FreeStyle Libre historical CGM readings
- User-initiated glucose scans
- Insulin pump records
- Carbohydrate entries
- Fitbit heart rate, steps, calories, and sleep measurements

Only raw FreeStyle glucose data are used in the primary project.

### Modeling cohort

- 25 participant folders discovered
- 24 parseable FreeStyle exports
- 23 participants with usable historical CGM
- 22 participants with both historical CGM and user-initiated scans

The paired continuous-versus-intermittent comparison uses the common 22-participant cohort.

## Why raw glucose data are required

The HUPA-UCM release includes preprocessed five-minute glucose tables, but the audit found extensive interpolation.

Across the 23 dense-cohort participants:
- approximately **90.5%** of preprocessed rows did not directly match raw CGM timestamps
- **3,308** preprocessed values below 70 mg/dL had no nearby raw low reading

Therefore:
- labels are constructed from raw historical CGM
- dense inputs use raw historical CGM
- sparse inputs use raw user-initiated scans
- preprocessed glucose is excluded from the primary model

## Leakage prevention

Longitudinal healthcare data can produce misleadingly high performance when nearby windows or the same participant appear in both training and testing.

This project prevents leakage by:
- grouping all windows from one participant into the same fold
- using observations strictly before prediction time
- defining outcomes only from the future label window
- reusing identical folds across every model
- avoiding participant ID as a feature
- excluding interpolated preprocessed glucose
- using a 30-minute stride to reduce near-duplicate windows

## Features

### Dense XGBoost features

Computed from the previous 4 hours of historical CGM:
- latest glucose, mean and median, minimum and maximum, standard deviation, range
- linear trend and changes over 15, 30, 60, and 120 minutes
- proportion of readings below 70, 80, and 90 mg/dL
- missing-slot count, time since last valid reading, cyclic time-of-day features

### Sparse XGBoost features

Computed from scans during the previous 6 hours:
- most recent scan value and age, number of scans
- mean, minimum, maximum, and standard deviation
- difference, time, and slope between the last two scans
- no-scan and one-scan indicators

Windows with no prior scan are retained rather than silently removed.

### GRU input

- 16 time steps at 15-minute resolution
- glucose-value channel and observation-mask channel

## Models

### XGBoost

XGBoost is the primary model because it handles nonlinear temporal features, performs well on small structured datasets, supports class weighting, and can be interpreted using SHAP.

### GRU

The GRU is intentionally small: one recurrent layer, 16 time steps, small hidden dimension, dropout, class-weighted loss, early stopping, and fixed random seeds. The goal is a controlled comparison between engineered tabular features and direct sequence modeling, not state-of-the-art performance.

## Evaluation metrics

The positive class represents approximately 15% of eligible prediction windows, so accuracy is not the primary metric.

**Primary metric:** Area under the precision-recall curve (AUPRC)

**Secondary metrics:** hypoglycemia recall, precision, F1 score, AUROC, specificity, confusion matrix

Results are reported using pooled out-of-fold predictions and participant-level summaries.

## Sensitivity analyses

### Excluding dominant participants

HUPA0027P and HUPA0028P contribute approximately 67% of identified hypoglycemic episodes. The primary XGBoost comparison is repeated without these two participants.

### Scan availability

Sparse-model performance is reported separately for all eligible windows, windows with at least one prior scan, and windows with no prior scan.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/kabJhai/glucose-gap.git
cd glucose-gap
```

### 2. Download the dataset

Download the [HUPA-UCM Diabetes Dataset](https://data.mendeley.com/datasets/3hbcscwz44/1) and extract it inside the project root:

```text
glucose-gap/
└── HUPA-UCM Diabetes Dataset/
    ├── Raw_Data/
    └── Preprocessed/
```

### 3. Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows: `.venv\Scripts\activate`

On macOS, install OpenMP if XGBoost requires it: `brew install libomp`

### 4. Run the pipeline

**Step 1: Feasibility audit** (~8-9 min)

```bash
python feasibility_audit/data_audit.py
```

**Step 2: Modeling experiments**

```bash
python -m modeling.train --skip-gru    # primary comparison (faster)
python -m modeling.train               # includes GRU experiment
```

### 5. Verify results

```bash
python scripts/verify_results.py
```

### 6. Run inference (deployed alert prototype)

Training also writes **deployment artifacts** under `modeling_outputs/artifacts/`:

| File | Purpose |
|------|---------|
| `dense_xgb.joblib` | Continuous CGM alert model + imputer + threshold |
| `sparse_xgb.joblib` | Intermittent scan alert model |
| `dense_gru.pt` | Optional sequence model (if GRU training ran) |
| `artifact_manifest.json` | Training metadata and disclaimer |

Score hypoglycemia risk for a participant (requires HUPA raw data on disk):

```bash
# All eligible 30-min windows for one participant
python -m modeling.predict --participant HUPA0001P

# Single prediction time (alert for next 2 hours)
python -m modeling.predict --participant HUPA0001P --at "2018-10-01 14:30:00"

# Write CSV alerts
python -m modeling.predict --participant HUPA0001P --output alerts.csv

# Include GRU arm
python -m modeling.predict --participant HUPA0001P --models dense,sparse,gru
```

**Output columns:** `risk_dense`, `alert_dense`, `risk_sparse`, `alert_sparse` (probability 0–1 and binary alert at tuned threshold).

**Inputs at inference time:** the pipeline reads the same FreeStyle CSV exports as training, builds a 4 h CGM history (dense) and 6 h scan history (sparse) strictly before each `prediction_time`, then scores the next **2 h** hypoglycemia risk.

This is a **deployable research prototype**, not a clinically validated alert system. See [Responsible use](#responsible-use).

## Bring your own dataset

Glucose Gap is not locked to HUPA-UCM. Any dataset with **continuous CGM** plus **intermittent scans** from the same sensor can use the same pipeline via a dataset profile.

### Supported layouts

| Layout | Config | Expected structure |
|--------|--------|-------------------|
| `hupa_ucm` (default) | `dataset_config.hupa.json` | `Raw_Data/<participant>/free_style_sensor/*.csv` |
| `canonical` | `dataset_config.example.json` | `participants/<participant>/glucose.csv` |

### Canonical `glucose.csv` schema

One file per participant:

```csv
timestamp,record_type,glucose_mg_dl
2018-06-19 17:19:00,0,142
2018-06-19 17:34:00,0,138
2018-06-19 17:40:00,1,135
```

| Column | Meaning |
|--------|---------|
| `timestamp` | ISO datetime |
| `record_type` | `0` = continuous CGM, `1` = user-initiated scan |
| `glucose_mg_dl` | Glucose in mg/dL |

Alternative: separate `historical_glucose_mg_dl` and `scan_glucose_mg_dl` columns (FreeStyle-style).

### Configure and validate

```bash
# Copy and edit the example profile
cp dataset_config.example.json my_dataset_config.json

# Point at your data root
python scripts/validate_dataset.py --dataset-config my_dataset_config.json --dataset-root data/my_cohort

# Train on your cohort
python -m modeling.train --dataset-config my_dataset_config.json --dataset-root data/my_cohort

# Inference
python -m modeling.predict --dataset-config my_dataset_config.json --participant P001
```

Environment overrides (optional):

```bash
export GLUCOSE_GAP_DATASET_CONFIG=my_dataset_config.json
export GLUCOSE_GAP_DATASET_ROOT=/path/to/cohort
```

### Convert HUPA to canonical (template for other exports)

```bash
python scripts/export_canonical.py --output data/exported_hupa
python -m modeling.train --dataset-config dataset_config.example.json --dataset-root data/exported_hupa
```

Set `cohort.exclude_sparse_no_scan` and `cohort.sensitivity_exclude` in your config for cohort rules on non-HUPA data.

## Reproducibility

The pipeline uses pinned dependencies, a fixed random seed (42), saved participant folds, inspectable intermediate CSVs, reference verification targets, and a run manifest written by `modeling/train.py`.

### Expected artifact checks

A successful run should generate:
- `paired_windows.csv`
- `fold_assignments.csv`
- `dense_features.csv`
- `sparse_features.csv`
- `model_metrics.csv`
- `participant_metrics.csv`
- `oof_predictions.csv`
- `modeling_results.md`
- `run_manifest.json`
- `artifacts/dense_xgb.joblib` (deployment model)
- `artifacts/sparse_xgb.joblib`
- `artifacts/artifact_manifest.json`

The paired window table should contain approximately **1,260** eligible prediction windows, **190** positive windows, and **22** participants.

## Repository structure

```text
glucose-gap/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── dataset/                    # dataset adapters (HUPA + canonical layout)
├── dataset_config.hupa.json    # default HUPA profile
├── dataset_config.example.json # template for other CGM datasets
├── modeling/                   # windows, features, CV, train, predict, GRU
│   ├── train.py                # cross-validation + save deployment artifacts
│   └── predict.py              # inference / alert scoring
├── scripts/
│   └── verify_results.py
├── tests/
├── tutorial/                   # walkthrough, slides, verification targets
└── modeling_outputs/           # generated and ignored
```

## Documentation

- [`tutorial/TUTORIAL.md`](tutorial/TUTORIAL.md): how the pipeline works, step by step
- [`tutorial/PRESENTATION.md`](tutorial/PRESENTATION.md): slide outline and speaker notes
- [`tutorial/verification_targets.json`](tutorial/verification_targets.json): reference metrics for `scripts/verify_results.py`

## Methods reference

Locked experimental design: [`feasibility_audit/feasibility_report.md`](feasibility_audit/feasibility_report.md)

## Methodological tests

Run design validation tests (no dataset required):

```bash
python -m unittest discover -s tests -v
```

The tests validate:
- dense and sparse models use identical prediction timestamps
- scan observations occur strictly before prediction time
- future glucose values never appear in features
- participants never cross CV folds
- fold assignments are deterministic
- sparse windows with no scans are retained and flagged

## Limitations

- The primary paired cohort contains only 22 participants.
- Hypoglycemic episodes are concentrated in a small number of participants.
- The dataset represents people with Type 1 diabetes using FreeStyle Libre.
- User-initiated scans are not equivalent to finger-prick measurements.
- No external dataset was used for validation.
- This is a retrospective educational analysis, not a clinically validated alert system.
- Insulin, meals, Fitbit activity, and sleep are excluded from version one.

## Future work

- **Clinical validation:** external cohorts, prospective evaluation, calibration for real alerts.
- **Model accuracy:** richer features (insulin, carbs), personalized models, longer horizons.
- **Production deployment:** REST API wrapper, streaming CGM ingestion, FHIR integration.
- Evaluate hyperglycemia prediction.
- Compare multiple prediction horizons.
- Model irregular scan sequences with masking or time-aware neural networks.
- Study realistic finger-prick schedules separately from user scan behavior.

## Citing and extending this work

This project is open source (MIT). If you fork, extend, or build on the Glucose Gap concept:

1. **Cite or link** the repository: https://github.com/kabJhai/glucose-gap
2. **Cite the dataset** when using HUPA-UCM (see [Dataset citation](#dataset-citation))
3. **Fork and advance** — swap models, add features, validate clinically; keep attribution in README and `artifact_manifest.json`
4. **Do not** present research-prototype alerts as clinical products without proper validation

Example attribution line:

> Based on [Glucose Gap](https://github.com/kabJhai/glucose-gap): measuring predictive loss between continuous CGM and intermittent glucose scans.

## Responsible use

This repository is intended for education and research.

The models are not validated for diagnosis, treatment, insulin dosing, or real-time clinical decision-making. Model predictions should not be used to guide patient care.

## Dataset citation

Hidalgo, J. Ignacio; Alvarado, Jorge; Botella, Marta; Aramendi, Aranzazu; Velasco, J. Manuel; Garnica, Oscar.  
"HUPA-UCM Diabetes Dataset." Mendeley Data, Version 1, 2024.  
DOI: [10.17632/3hbcscwz44.1](https://doi.org/10.17632/3hbcscwz44.1)

Data paper: [Campos et al., *Data in Brief*, 2024](https://doi.org/10.1016/j.dib.2024.110559)

## License

The project code is released under the [MIT License](LICENSE).

The HUPA-UCM dataset is not distributed in this repository and remains governed by its original CC BY 4.0 license.
