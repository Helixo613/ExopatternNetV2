"""
Model training pipeline for anomaly detection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

from .preprocessing import LightCurvePreprocessor
from .models import AnomalyDetector

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training of anomaly detection models.
    """

    def __init__(self, model_save_dir: str = 'backend/models'):
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = LightCurvePreprocessor()
        self.detector = None

    def train_from_files(self, file_paths: List[str],
                        contamination: float = 0.1,
                        window_size: int = 50) -> AnomalyDetector:
        """
        Train anomaly detector from multiple light curve files.

        Args:
            file_paths: List of paths to light curve files (FITS or CSV)
            contamination: Expected proportion of anomalies
            window_size: Window size for feature extraction

        Returns:
            Trained AnomalyDetector
        """
        from backend.data import LightCurveLoader

        logger.info(f"Training on {len(file_paths)} light curve files...")

        # Load and preprocess all files
        all_features = []

        loader = LightCurveLoader()

        for file_path in file_paths:
            try:
                # Load data
                df = loader.load_file(file_path)
                logger.info(f"Loaded {file_path}: {len(df)} points")

                # Preprocess
                df_processed = self.preprocessor.preprocess(df, normalize=True)

                # Extract features
                features = self.preprocessor.extract_features(df_processed, window_size)
                all_features.append(features)

                logger.info(f"Extracted {len(features)} feature windows")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue

        if not all_features:
            raise ValueError("No valid training data extracted")

        # Combine all features
        combined_features = np.vstack(all_features)
        logger.info(f"Total training samples: {len(combined_features)}")

        # Train detector
        self.detector = AnomalyDetector(contamination=contamination)
        self.detector.fit(combined_features)

        # Save model
        self.save_model()

        return self.detector

    def train_from_dataframe(self, df: pd.DataFrame,
                            contamination: float = 0.1,
                            window_size: int = 50) -> AnomalyDetector:
        """
        Train anomaly detector from a single DataFrame.

        Args:
            df: DataFrame with time, flux columns
            contamination: Expected proportion of anomalies
            window_size: Window size for feature extraction

        Returns:
            Trained AnomalyDetector
        """
        logger.info(f"Training on single light curve: {len(df)} points")

        # Preprocess
        df_processed = self.preprocessor.preprocess(df, normalize=True)

        # Extract features
        features = self.preprocessor.extract_features(df_processed, window_size)
        logger.info(f"Extracted {len(features)} feature windows")

        # Train detector
        self.detector = AnomalyDetector(contamination=contamination)
        self.detector.fit(features)

        # Save model
        self.save_model()

        return self.detector

    def train_with_synthetic_data(self, n_samples: int = 1000,
                                  contamination: float = 0.1) -> AnomalyDetector:
        """
        Train on synthetic light curve data for initial model.

        This is useful for creating a baseline model when no training data is available.

        Args:
            n_samples: Number of synthetic light curves to generate
            contamination: Proportion of anomalies to inject

        Returns:
            Trained AnomalyDetector
        """
        logger.info(f"Generating {n_samples} synthetic light curves...")

        all_features = []

        for i in range(n_samples):
            # Generate synthetic light curve
            df = self._generate_synthetic_lightcurve(
                n_points=1000,
                has_anomaly=(i < n_samples * contamination)
            )

            # Preprocess
            df_processed = self.preprocessor.preprocess(df, normalize=True)

            # Extract features
            features = self.preprocessor.extract_features(df_processed, window_size=50)
            all_features.append(features)

        # Combine features
        combined_features = np.vstack(all_features)
        logger.info(f"Total synthetic samples: {len(combined_features)}")

        # Train detector
        self.detector = AnomalyDetector(contamination=contamination)
        self.detector.fit(combined_features)

        # Save model
        self.save_model('default_model')

        return self.detector

    def _generate_synthetic_lightcurve(self, n_points: int = 1000,
                                      has_anomaly: bool = False) -> pd.DataFrame:
        """
        Generate a synthetic stellar light curve.

        Creates realistic light curves with optional anomalies (transits, flares, or noise).
        """
        # Time array
        time = np.linspace(0, 100, n_points)

        # Base flux (constant star)
        flux = np.ones(n_points) * 1000

        # Add stellar variability (noise)
        flux += np.random.normal(0, 5, n_points)

        # Add slow trend (stellar activity)
        trend = 10 * np.sin(2 * np.pi * time / 50)
        flux += trend

        # Add anomalies if requested
        if has_anomaly:
            anomaly_type = np.random.choice(['transit', 'flare', 'outlier'])

            if anomaly_type == 'transit':
                # Add transit-like dip
                transit_start = np.random.randint(100, n_points - 200)
                transit_duration = np.random.randint(20, 50)
                depth = np.random.uniform(10, 50)

                # Create transit shape (box or trapezoid)
                for i in range(transit_start, transit_start + transit_duration):
                    if i < len(flux):
                        flux[i] -= depth

            elif anomaly_type == 'flare':
                # Add flare (spike)
                flare_start = np.random.randint(100, n_points - 100)
                flare_duration = np.random.randint(5, 20)
                amplitude = np.random.uniform(50, 150)

                for i in range(flare_start, flare_start + flare_duration):
                    if i < len(flux):
                        # Exponential decay
                        decay = np.exp(-(i - flare_start) / 5)
                        flux[i] += amplitude * decay

            else:  # outlier
                # Add random outliers
                n_outliers = np.random.randint(5, 15)
                outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
                outlier_values = np.random.normal(0, 100, n_outliers)
                flux[outlier_indices] += outlier_values

        # Add flux errors
        flux_err = np.abs(np.random.normal(0, 2, n_points))

        return pd.DataFrame({
            'time': time,
            'flux': flux,
            'flux_err': flux_err
        })

    def save_model(self, name: str = 'trained_model'):
        """Save the trained model."""
        if self.detector is None:
            raise ValueError("No model to save")

        save_path = self.model_save_dir / name
        self.detector.save_model(str(save_path))
        logger.info(f"Model saved to {save_path}")

    def load_model(self, name: str = 'trained_model') -> AnomalyDetector:
        """Load a trained model."""
        load_path = self.model_save_dir / name

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        self.detector = AnomalyDetector()
        self.detector.load_model(str(load_path))
        logger.info(f"Model loaded from {load_path}")

        return self.detector

    def evaluate_model(self, df: pd.DataFrame, window_size: int = 50) -> Dict:
        """
        Evaluate model performance on a light curve.

        Args:
            df: DataFrame with time, flux columns
            window_size: Window size for feature extraction

        Returns:
            Dictionary with evaluation metrics
        """
        if self.detector is None or not self.detector.is_fitted:
            raise ValueError("Model must be trained before evaluation")

        # Preprocess
        df_processed = self.preprocessor.preprocess(df, normalize=True)

        # Extract features
        features = self.preprocessor.extract_features(df_processed, window_size)

        # Predict
        predictions, scores = self.detector.predict_with_scores(features)

        # Calculate statistics
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)

        return {
            'n_windows': len(predictions),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'mean_score': float(np.mean(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
        }
