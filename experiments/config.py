"""
Experiment configuration constants and hyperparameter grids.
"""

# Reproducibility
RANDOM_SEED = 42

# Cross-validation
N_CV_FOLDS = 5
CV_STRATEGY = 'stratified'

# Bootstrap
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

# Default feature groups
DEFAULT_FEATURE_GROUPS = ['statistical', 'frequency', 'wavelet', 'autocorrelation', 'shape']

# Feature ablation order (cumulative)
ABLATION_ORDER = [
    ['statistical'],
    ['statistical', 'frequency'],
    ['statistical', 'frequency', 'wavelet'],
    ['statistical', 'frequency', 'wavelet', 'autocorrelation'],
    ['statistical', 'frequency', 'wavelet', 'autocorrelation', 'shape'],
]

ABLATION_LABELS = [
    'Statistical',
    '+ Frequency',
    '+ Wavelet',
    '+ Autocorrelation',
    '+ Shape',
]

# Models to compare
CLASSICAL_MODELS = ['isolation_forest', 'lof', 'ocsvm', 'dbscan', 'ensemble']
DEEP_MODELS = ['cnn_autoencoder', 'lstm_autoencoder']
ALL_MODELS = CLASSICAL_MODELS + DEEP_MODELS

# Default hyperparameters
DEFAULT_CONTAMINATION = 0.1
DEFAULT_WINDOW_SIZE = 50

# Hyperparameter grids for sensitivity analysis
HYPERPARAM_GRIDS = {
    'contamination': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
    'window_size': [10, 20, 30, 50, 75, 100, 150],
    'n_estimators': [50, 100, 200, 500],
}

# Output paths
RESULTS_DIR = 'results'
FIGURES_DIR = 'results/figures'
TABLES_DIR = 'results/tables'
RAW_DIR = 'results/raw'
