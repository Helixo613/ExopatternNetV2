"""
Model registry for anomaly detection models.

Provides a unified interface to create models by name,
and load pre-trained models from disk.
"""

from typing import Dict, Callable, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


def _if_factory(**kwargs):
    from .baselines import IsolationForestModel
    return IsolationForestModel(**kwargs)


def _lof_factory(**kwargs):
    from .baselines import LOFModel
    return LOFModel(**kwargs)


def _ocsvm_factory(**kwargs):
    from .baselines import OneClassSVMModel
    return OneClassSVMModel(**kwargs)


def _dbscan_factory(**kwargs):
    from .baselines import DBSCANModel
    return DBSCANModel(**kwargs)


def _ensemble_factory(**kwargs):
    from .baselines import EnsembleIFLOF
    return EnsembleIFLOF(**kwargs)


def _bls_factory(**kwargs):
    from .baselines import BLSDetector
    return BLSDetector(**kwargs)


def _cnn_ae_factory(**kwargs):
    from .deep_models import Conv1DAutoencoder
    return Conv1DAutoencoder(**kwargs)


def _lstm_ae_factory(**kwargs):
    from .deep_models import LSTMAutoencoder
    return LSTMAutoencoder(**kwargs)


# Registry mapping model names to factory functions
MODEL_REGISTRY: Dict[str, Callable] = {
    'isolation_forest': _if_factory,
    'lof': _lof_factory,
    'ocsvm': _ocsvm_factory,
    'dbscan': _dbscan_factory,
    'ensemble': _ensemble_factory,
    'bls': _bls_factory,
    'cnn_autoencoder': _cnn_ae_factory,
    'lstm_autoencoder': _lstm_ae_factory,
}

# Human-readable names
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    'isolation_forest': 'Isolation Forest',
    'lof': 'Local Outlier Factor',
    'ocsvm': 'One-Class SVM',
    'dbscan': 'DBSCAN',
    'ensemble': 'Ensemble (IF+LOF)',
    'bls': 'Box Least Squares',
    'cnn_autoencoder': 'CNN Autoencoder',
    'lstm_autoencoder': 'LSTM Autoencoder',
}

# Models that don't require GPU
CLASSICAL_MODELS = ['isolation_forest', 'lof', 'ocsvm', 'dbscan', 'ensemble', 'bls']

# Models that benefit from GPU
DEEP_MODELS = ['cnn_autoencoder', 'lstm_autoencoder']


def get_model(name: str, **kwargs):
    """
    Create a model instance by name.

    Args:
        name: Model identifier (see list_models())
        **kwargs: Model-specific parameters (contamination, random_state, etc.)

    Returns:
        Model instance with fit/predict/score_samples interface
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


def list_models(include_deep: bool = True) -> List[str]:
    """List available model names."""
    if include_deep:
        return list(MODEL_REGISTRY.keys())
    return CLASSICAL_MODELS.copy()


def get_display_name(name: str) -> str:
    """Get human-readable display name for a model."""
    return MODEL_DISPLAY_NAMES.get(name, name)


# Mapping from model name to artifact filename (without extension)
_ARTIFACT_FILENAMES: Dict[str, str] = {
    'isolation_forest': 'isolation_forest',
    'lof': 'lof',
    'ocsvm': 'ocsvm',
    'dbscan': 'dbscan',
    'ensemble': 'ensemble',
    'cnn_autoencoder': 'cnn_autoencoder',
    'lstm_autoencoder': 'lstm_autoencoder',
}


def load_pretrained(name: str, models_dir: str) -> Optional[object]:
    """
    Load a pre-trained model from disk.

    Classical models (.pkl) are loaded with joblib.
    Deep models (.h5) are instantiated and loaded via their .load() method.
    If the model file is not found, returns None (allows fallback).

    Args:
        name: Model identifier (e.g. 'isolation_forest', 'cnn_autoencoder')
        models_dir: Path to directory containing model artifacts

    Returns:
        Fitted model ready for .predict()/.score_samples(), or None
    """
    if name not in _ARTIFACT_FILENAMES:
        return None

    base = _ARTIFACT_FILENAMES[name]

    # Classical models: .pkl files
    if name in CLASSICAL_MODELS and name != 'bls':
        pkl_path = os.path.join(models_dir, f"{base}.pkl")
        if not os.path.exists(pkl_path):
            logger.warning(f"Pre-trained model not found: {pkl_path}")
            return None
        import joblib
        model = joblib.load(pkl_path)
        logger.info(f"Loaded pre-trained classical model: {name} from {pkl_path}")
        return model

    # Deep models: .h5 files
    if name in DEEP_MODELS:
        h5_path = os.path.join(models_dir, f"{base}.h5")
        if not os.path.exists(h5_path):
            logger.warning(f"Pre-trained model not found: {h5_path}")
            return None

        # Load meta JSON for threshold/params
        import json
        meta_path = os.path.join(models_dir, f"{base}_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        # Instantiate the model class
        model = MODEL_REGISTRY[name]()

        # Check for params.npz (saved by model.save())
        npz_path = f"{h5_path}_params.npz"
        if os.path.exists(npz_path):
            model.load(h5_path)
        else:
            # Load .h5 weights only; set threshold from meta JSON.
            # mean/std will be computed from input data at predict time.
            import numpy as np
            from tensorflow import keras
            model.model = keras.models.load_model(h5_path, compile=False)
            model.threshold = meta.get('threshold', 0.1)
            if hasattr(model, 'seq_length') and 'seq_length' in meta:
                model.seq_length = meta['seq_length']
            model._mean = None
            model._std = None
            model.is_fitted = True

        logger.info(f"Loaded pre-trained deep model: {name} from {h5_path}")
        return model

    return None
