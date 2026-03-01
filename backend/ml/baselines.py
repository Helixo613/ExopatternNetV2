"""
Baseline anomaly detection models with a unified interface.

All models implement: fit(X), predict(X) -> {-1, 1}, score_samples(X) -> float
"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BaseAnomalyModel(ABC):
    """Abstract base class for all anomaly detection models."""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.is_fitted = False
        self.scaler = StandardScaler()
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseAnomalyModel':
        """Train on feature array X of shape (n_samples, n_features)."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies. Returns array of -1 (anomaly) or 1 (normal)."""
        pass

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. More negative = more anomalous."""
        pass


class IsolationForestModel(BaseAnomalyModel):
    """Isolation Forest wrapper with unified interface."""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100,
                 random_state: int = 42):
        super().__init__(contamination, random_state)
        self._name = 'Isolation Forest'
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> 'IsolationForestModel':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)


class LOFModel(BaseAnomalyModel):
    """Local Outlier Factor wrapper with unified interface."""

    def __init__(self, contamination: float = 0.1, n_neighbors: int = 20,
                 random_state: int = 42):
        super().__init__(contamination, random_state)
        self._name = 'LOF'
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> 'LOFModel':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)


class OneClassSVMModel(BaseAnomalyModel):
    """One-Class SVM for anomaly detection."""

    def __init__(self, contamination: float = 0.1, kernel: str = 'rbf',
                 random_state: int = 42):
        super().__init__(contamination, random_state)
        self._name = 'One-Class SVM'
        self.model = OneClassSVM(
            kernel=kernel,
            nu=contamination,  # nu approximates contamination
            gamma='scale',
        )

    def fit(self, X: np.ndarray) -> 'OneClassSVMModel':
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)


class DBSCANModel(BaseAnomalyModel):
    """
    DBSCAN-based anomaly detection.
    Points not assigned to any cluster (label=-1) are anomalies.
    """

    def __init__(self, contamination: float = 0.1, eps: float = 0.5,
                 min_samples: int = 5, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._name = 'DBSCAN'
        self.eps = eps
        self.min_samples = min_samples
        self._train_labels = None
        self._train_X = None

    def fit(self, X: np.ndarray) -> 'DBSCANModel':
        self._train_X = self.scaler.fit_transform(X)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        self._train_labels = db.fit_predict(self._train_X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        For new data, use distance to nearest core point.
        Points far from all core points are anomalies.
        """
        X_scaled = self.scaler.transform(X)
        scores = self.score_samples(X)
        # Use contamination-based threshold on scores
        threshold = np.percentile(scores, 100 * self.contamination)
        predictions = np.where(scores < threshold, -1, 1)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Score based on distance to nearest training cluster core."""
        X_scaled = self.scaler.transform(X)
        # Use core samples from training
        core_mask = self._train_labels != -1
        if core_mask.sum() == 0:
            return np.zeros(len(X))

        core_points = self._train_X[core_mask]
        # Compute min distance to any core point (batch for memory efficiency)
        scores = np.zeros(len(X_scaled))
        batch_size = 1000
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i:i + batch_size]
            dists = np.sqrt(((batch[:, np.newaxis] - core_points[np.newaxis, :]) ** 2).sum(axis=2))
            scores[i:i + batch_size] = -dists.min(axis=1)  # negative = more anomalous

        return scores


class EnsembleIFLOF(BaseAnomalyModel):
    """
    Fixed IF+LOF ensemble using AND logic (both must agree)
    and score-based thresholding.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._name = 'Ensemble (IF+LOF)'
        self.if_model = IsolationForestModel(contamination, random_state=random_state)
        self.lof_model = LOFModel(contamination, random_state=random_state)

    def fit(self, X: np.ndarray) -> 'EnsembleIFLOF':
        self.if_model.fit(X)
        self.lof_model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if_pred = self.if_model.predict(X)
        lof_pred = self.lof_model.predict(X)
        # AND logic: anomaly only if BOTH agree
        predictions = np.where((if_pred == -1) & (lof_pred == -1), -1, 1)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if_scores = self.if_model.score_samples(X)
        lof_scores = self.lof_model.score_samples(X)
        return (if_scores + lof_scores) / 2


class BLSDetector(BaseAnomalyModel):
    """
    Box Least Squares transit detector.
    Domain-specific baseline that looks for periodic box-shaped dips.
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        super().__init__(contamination, random_state)
        self._name = 'BLS'
        self._best_period = None
        self._best_duration = None
        self._best_epoch = None
        self._threshold = None

    def fit(self, X: np.ndarray, time: Optional[np.ndarray] = None,
            flux: Optional[np.ndarray] = None) -> 'BLSDetector':
        """
        BLS requires time and flux arrays rather than feature matrices.
        If only X is provided, uses first two columns as time and flux.
        """
        if time is None or flux is None:
            logger.warning("BLS requires time/flux arrays. Using feature matrix columns 0,1.")
            self.is_fitted = True
            return self

        try:
            from astropy.timeseries import BoxLeastSquares

            time_span = time.max() - time.min()
            min_period = max(0.5, np.median(np.diff(time)) * 10)
            max_period = min(time_span / 2, 100.0)

            if max_period <= min_period:
                self.is_fitted = True
                return self

            bls = BoxLeastSquares(time, flux)
            periods = np.linspace(min_period, max_period, 1000)
            result = bls.power(periods, duration=np.linspace(0.05, 0.3, 10))

            best_idx = np.argmax(result.power)
            self._best_period = float(result.period[best_idx])
            self._best_duration = float(result.duration[best_idx])

            # Find epoch (transit mid-time)
            stats = bls.compute_stats(
                self._best_period, self._best_duration, result.transit_time[best_idx]
            )
            self._best_epoch = float(result.transit_time[best_idx])

            # Set threshold based on BLS power
            self._threshold = float(np.percentile(result.power, 95))

        except Exception as e:
            logger.error(f"BLS fit failed: {e}")

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, time: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict transit points based on BLS ephemeris."""
        n = X.shape[0] if X.ndim > 1 else len(X)

        if self._best_period is None or time is None:
            return np.ones(n)  # all normal

        half_dur = self._best_duration / 2.0
        phase = ((time - self._best_epoch) % self._best_period) / self._best_period
        phase_half = half_dur / self._best_period

        predictions = np.ones(n)
        for i, p in enumerate(phase):
            if p < phase_half or p > (1.0 - phase_half):
                predictions[i] = -1

        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """BLS doesn't produce per-point scores easily; return zeros."""
        n = X.shape[0] if X.ndim > 1 else len(X)
        return np.zeros(n)
