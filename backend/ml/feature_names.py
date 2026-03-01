"""
Feature name registry for mapping feature groups to names and indices.
Used for ablation studies and feature importance analysis.
"""

# Feature groups and their names (in extraction order)
FEATURE_GROUPS = {
    'statistical': [
        'mean', 'std', 'median', 'min', 'max', 'ptp',
        'skewness', 'kurtosis', 'slope', 'intercept',
        'mad', 'rms', 'mean_abs_diff', 'max_abs_diff',
    ],
    'frequency': [
        'fft_amp_1', 'fft_amp_2', 'fft_amp_3',
        'dominant_freq', 'spectral_entropy',
        'spectral_centroid', 'spectral_rolloff',
    ],
    'wavelet': [
        'cwt_energy_scale1', 'cwt_energy_scale2', 'cwt_energy_scale3',
        'cwt_energy_scale4', 'cwt_energy_scale5',
        'wavelet_entropy', 'max_energy_scale',
    ],
    'autocorrelation': [
        'acf_lag1', 'acf_lag5', 'acf_lag10',
        'acf_first_zero', 'acf_decay_rate',
    ],
    'shape': [
        'asymmetry', 'flatness', 'edge_gradient_ratio',
        'min_position', 'beyond_1sigma_frac',
    ],
}

# Global features (not per-window, computed once per light curve)
GLOBAL_FEATURE_GROUPS = {
    'periodogram': [
        'ls_max_power', 'ls_dominant_period', 'ls_second_power',
        'ls_power_ratio', 'bls_power', 'bls_period', 'bls_duration',
    ],
}

# Default groups used when no feature_groups parameter is specified
DEFAULT_GROUPS = ['statistical', 'frequency', 'wavelet', 'autocorrelation', 'shape']

# All feature names in order
ALL_WINDOW_FEATURES = []
for group in DEFAULT_GROUPS:
    ALL_WINDOW_FEATURES.extend(FEATURE_GROUPS[group])


def get_feature_names(groups=None):
    """Get ordered list of feature names for given groups."""
    if groups is None:
        groups = DEFAULT_GROUPS
    names = []
    for g in groups:
        if g in FEATURE_GROUPS:
            names.extend(FEATURE_GROUPS[g])
        elif g in GLOBAL_FEATURE_GROUPS:
            names.extend(GLOBAL_FEATURE_GROUPS[g])
    return names


def get_feature_indices(group_name):
    """Get start and end indices for a feature group within the full feature vector."""
    offset = 0
    for g in DEFAULT_GROUPS:
        if g == group_name:
            return offset, offset + len(FEATURE_GROUPS[g])
        offset += len(FEATURE_GROUPS[g])
    raise ValueError(f"Unknown group: {group_name}")


def get_n_features(groups=None):
    """Get total number of features for given groups."""
    return len(get_feature_names(groups))
