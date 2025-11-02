"""
Anomaly detection models for stellar light curves.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Multi-algorithm anomaly detection for stellar light curves.

    Combines multiple detection methods:
    - Isolation Forest (primary)
    - Local Outlier Factor
    - Statistical threshold detection
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies in dataset
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state

        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1
        )

        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination,
            novelty=True,  # Allow prediction on new data
            n_jobs=-1
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, features: np.ndarray) -> 'AnomalyDetector':
        """
        Train anomaly detection models.

        Args:
            features: Feature array of shape (n_samples, n_features)

        Returns:
            self
        """
        logger.info(f"Training anomaly detector on {len(features)} samples...")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Fit Isolation Forest
        self.isolation_forest.fit(features_scaled)
        logger.info("Isolation Forest trained")

        # Fit LOF
        self.lof.fit(features_scaled)
        logger.info("Local Outlier Factor trained")

        self.is_fitted = True
        logger.info("Anomaly detector training complete")

        return self

    def predict(self, features: np.ndarray, method: str = 'ensemble') -> np.ndarray:
        """
        Predict anomalies in light curve.

        Args:
            features: Feature array of shape (n_samples, n_features)
            method: Detection method - 'isolation_forest', 'lof', or 'ensemble'

        Returns:
            Binary array: 1 for anomaly, 0 for normal (or -1/1 for sklearn convention)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features_scaled = self.scaler.transform(features)

        if method == 'isolation_forest':
            predictions = self.isolation_forest.predict(features_scaled)

        elif method == 'lof':
            predictions = self.lof.predict(features_scaled)

        elif method == 'ensemble':
            # Combine predictions from both methods
            if_pred = self.isolation_forest.predict(features_scaled)
            lof_pred = self.lof.predict(features_scaled)

            # Anomaly if either method flags it (-1 in sklearn convention)
            predictions = np.where((if_pred == -1) | (lof_pred == -1), -1, 1)

        else:
            raise ValueError(f"Unknown method: {method}")

        return predictions

    def predict_with_scores(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies with anomaly scores.

        Returns:
            predictions: Binary predictions (1/-1)
            scores: Anomaly scores (more negative = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        features_scaled = self.scaler.transform(features)

        # Get anomaly scores (more negative = more anomalous)
        if_scores = self.isolation_forest.score_samples(features_scaled)
        lof_scores = self.lof.score_samples(features_scaled)

        # Combine scores (average)
        combined_scores = (if_scores + lof_scores) / 2

        # Get predictions
        predictions = self.predict(features, method='ensemble')

        return predictions, combined_scores

    def detect_point_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict:
        """
        Detect point anomalies using statistical methods.

        This is a quick method that doesn't require training.

        Args:
            df: DataFrame with time and flux
            threshold: Number of standard deviations for anomaly threshold

        Returns:
            Dictionary with anomaly information
        """
        flux = df['flux'].values
        time = df['time'].values

        # Calculate z-scores
        mean = np.mean(flux)
        std = np.std(flux)
        z_scores = np.abs((flux - mean) / std)

        # Find anomalies
        anomaly_mask = z_scores > threshold
        anomaly_indices = np.where(anomaly_mask)[0]

        # Categorize anomalies
        flux_diff = flux - mean
        dips = anomaly_indices[flux_diff[anomaly_indices] < 0]  # Below mean
        spikes = anomaly_indices[flux_diff[anomaly_indices] > 0]  # Above mean

        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_times': time[anomaly_indices].tolist(),
            'anomaly_fluxes': flux[anomaly_indices].tolist(),
            'z_scores': z_scores[anomaly_indices].tolist(),
            'n_anomalies': len(anomaly_indices),
            'n_dips': len(dips),
            'n_spikes': len(spikes),
            'dip_indices': dips.tolist(),
            'spike_indices': spikes.tolist(),
        }

    def detect_transit_events(self, df: pd.DataFrame, depth_threshold: float = 0.01,
                              duration_min: int = 3) -> List[Dict]:
        """
        Detect potential transit events (sustained flux decreases).

        Args:
            df: DataFrame with time and flux
            depth_threshold: Minimum relative flux decrease
            duration_min: Minimum number of consecutive points

        Returns:
            List of transit event dictionaries
        """
        flux = df['flux'].values
        time = df['time'].values

        # Normalize to median
        median_flux = np.median(flux)
        rel_flux = flux / median_flux

        # Find points below threshold
        below_threshold = rel_flux < (1 - depth_threshold)

        # Find consecutive sequences
        events = []
        in_event = False
        event_start = None

        for i, is_below in enumerate(below_threshold):
            if is_below and not in_event:
                # Start of event
                in_event = True
                event_start = i
            elif not is_below and in_event:
                # End of event
                event_end = i
                event_duration = event_end - event_start

                if event_duration >= duration_min:
                    event_flux = flux[event_start:event_end]
                    event_time = time[event_start:event_end]

                    events.append({
                        'start_idx': event_start,
                        'end_idx': event_end,
                        'duration': event_duration,
                        'start_time': float(event_time[0]),
                        'end_time': float(event_time[-1]),
                        'time_duration': float(event_time[-1] - event_time[0]),
                        'depth': float(1 - np.min(rel_flux[event_start:event_end])),
                        'mean_flux': float(np.mean(event_flux)),
                    })

                in_event = False

        logger.info(f"Detected {len(events)} potential transit events")
        return events

    def save_model(self, save_dir: str):
        """Save trained models to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save models
        joblib.dump(self.isolation_forest, save_path / 'isolation_forest.pkl')
        joblib.dump(self.lof, save_path / 'lof.pkl')
        joblib.dump(self.scaler, save_path / 'scaler.pkl')

        # Save metadata
        metadata = {
            'contamination': self.contamination,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
        }
        joblib.dump(metadata, save_path / 'metadata.pkl')

        logger.info(f"Model saved to {save_path}")

    def load_model(self, save_dir: str):
        """Load trained models from disk."""
        save_path = Path(save_dir)

        if not save_path.exists():
            raise FileNotFoundError(f"Model directory not found: {save_path}")

        # Load models
        self.isolation_forest = joblib.load(save_path / 'isolation_forest.pkl')
        self.lof = joblib.load(save_path / 'lof.pkl')
        self.scaler = joblib.load(save_path / 'scaler.pkl')

        # Load metadata
        metadata = joblib.load(save_path / 'metadata.pkl')
        self.contamination = metadata['contamination']
        self.random_state = metadata['random_state']
        self.is_fitted = metadata['is_fitted']

        logger.info(f"Model loaded from {save_path}")


class EnsembleAnomalyDetector:
    """
    Advanced ensemble combining multiple detection strategies.
    """

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.detectors = []
        self.weights = []

    def add_detector(self, detector, weight: float = 1.0):
        """Add a detector to the ensemble."""
        self.detectors.append(detector)
        self.weights.append(weight)

    def fit(self, features: np.ndarray):
        """Train all detectors."""
        for detector in self.detectors:
            detector.fit(features)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict using weighted voting.
        """
        if not self.detectors:
            raise ValueError("No detectors in ensemble")

        predictions = []
        for detector in self.detectors:
            pred = detector.predict(features)
            predictions.append(pred)

        # Weighted voting
        weighted_votes = np.zeros(len(features))
        for pred, weight in zip(predictions, self.weights):
            # Convert -1/1 to 0/1
            pred_binary = (pred == -1).astype(int)
            weighted_votes += pred_binary * weight

        # Threshold at half of total weight
        threshold = sum(self.weights) / 2
        final_predictions = np.where(weighted_votes > threshold, -1, 1)

        return final_predictions
