"""
Configuration for ExoPattern v3.1 paper experiments.

"Conformal Anomaly Ranking for Transit Candidate Prioritization in Kepler Light Curves"
"""

from backend.ml.injection import RADIUS_GRID_REARTH, PERIOD_GRID_DAYS

# -----------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------
METADATA_CSV    = 'data/labeled/metadata.csv'
LIGHTCURVE_DIR  = 'data/labeled/lightcurves'

# -----------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------
N_CV_FOLDS      = 5     # outer GroupKFold folds
CALIB_FRACTION  = 0.25  # 25% of non-test stars → calibration (30/120 for full set)

# -----------------------------------------------------------------------
# Pipeline defaults (must match BLUEPRINT D1–D6)
# -----------------------------------------------------------------------
WINDOW_SIZE        = 50
STRIDE             = 12
FLAG_QUANTILE      = 0.975      # 97.5th percentile composite threshold (blueprint D3)
GAP_TOLERANCE      = 2          # windows
MAX_EVENT_WINDOWS  = 20
EVENT_SCORE_METHOD = 'top3_mean'
TLS_ALPHA          = 0.7        # ranking = 0.7*composite + 0.3*consistency_novelty
CONTAMINATION      = 0.1
MAX_TRAIN_WINDOWS  = 200_000    # bound fold-fit RAM via proportional pre-subsampling

# -----------------------------------------------------------------------
# Conformal calibration
# -----------------------------------------------------------------------
ALPHA_LEVELS = (0.01, 0.05, 0.10)   # significance levels for reporting

# -----------------------------------------------------------------------
# Injection-recovery
# -----------------------------------------------------------------------
INJECTION_RADIUS_GRID  = RADIUS_GRID_REARTH     # R_Earth, 8 values
INJECTION_PERIOD_GRID  = PERIOD_GRID_DAYS        # days, 8 values
INJECTION_N_TRIALS     = 25                      # per grid cell (1,600 total)
INJECTION_RECOVERY_WINDOW_FACTOR = 1.5           # × transit duration

# -----------------------------------------------------------------------
# Ablation study variants (Experiment 3)
# -----------------------------------------------------------------------
TLS_ALPHA_ABLATION = [0.0, 0.3, 0.5, 0.7, 1.0]   # 0.0 = consistency only, 1.0 = composite only
EVENT_SCORE_ABLATION = ['max', 'top3_mean', 'length_penalized']

# -----------------------------------------------------------------------
# SHAP explainability (Experiment 4)
# -----------------------------------------------------------------------
SHAP_TOP_K = 10   # top-K candidates to explain

# -----------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------
RESULTS_DIR  = 'results/paper'
FIGURES_DIR  = 'results/paper/figures'
TABLES_DIR   = 'results/paper/tables'
RAW_DIR      = 'results/paper/raw'
