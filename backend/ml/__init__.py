"""
Machine learning module for anomaly detection in stellar light curves.
"""

from .preprocessing import LightCurvePreprocessor
from .models import AnomalyDetector, map_window_predictions_to_points
from .training import ModelTrainer
from .evaluation import AnomalyEvaluator
from .model_registry import get_model, list_models, get_display_name, MODEL_REGISTRY

__all__ = [
    'LightCurvePreprocessor', 'AnomalyDetector', 'ModelTrainer',
    'map_window_predictions_to_points', 'AnomalyEvaluator',
    'get_model', 'list_models', 'get_display_name', 'MODEL_REGISTRY',
]
