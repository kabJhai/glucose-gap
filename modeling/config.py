"""Locked modeling configuration (see feasibility_audit/feasibility_report.md §11)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = PROJECT_ROOT / "feasibility_audit"
DATASET_ROOT = PROJECT_ROOT / "HUPA-UCM Diabetes Dataset"
OUTPUT_DIR = PROJECT_ROOT / "modeling_outputs"

DENSE_HISTORY_H = 4
SPARSE_HISTORY_H = 6
HORIZON_H = 2
STRIDE_MIN = 30
MISSINGNESS_THRESHOLD = 0.20
HYPO_THRESHOLD = 70
CGM_SLOT_MIN = 15
DENSE_SEQ_LEN = 16  # 4h at 15-min resolution

COMMON_COHORT_EXCLUDE_SCANS = ["HUPA0015P"]
SENSITIVITY_EXCLUDE = ["HUPA0027P", "HUPA0028P"]
N_FOLDS = 5
RANDOM_SEED = 42

GRU_HIDDEN = 32
GRU_DROPOUT = 0.3
GRU_EPOCHS = 50
GRU_PATIENCE = 8
GRU_BATCH_SIZE = 64
GRU_LR = 1e-3

# Inner validation split (of the training folds) used for threshold tuning only.
INNER_VAL_FRACTION = 0.2
# Number of participant-level bootstrap resamples for the paired difference CI.
N_BOOTSTRAP = 2000
# Episode separation (minutes) for participant-level episode counts used in fold balancing.
EPISODE_SEPARATION_MIN = 30
