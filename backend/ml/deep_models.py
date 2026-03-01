"""
Deep learning autoencoder models for anomaly detection.

Anomalies are detected via high reconstruction error:
if the autoencoder can't reconstruct a sample well, it's anomalous.

Uses tensorflow/keras. Models:
- Conv1DAutoencoder: 1D-CNN autoencoder on feature windows
- LSTMAutoencoder: LSTM autoencoder on sequences of windows
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not available. Deep learning models disabled.")


class Conv1DAutoencoder:
    """
    1D Convolutional Autoencoder for anomaly detection.

    Architecture:
        Encoder: Conv1D(32) -> Conv1D(16) -> Conv1D(8) -> Flatten -> Dense(latent)
        Decoder: Dense -> Reshape -> Conv1DTranspose(8) -> Conv1DTranspose(16) -> Conv1DTranspose(32) -> output

    Anomaly score = reconstruction error (MSE per sample).
    """

    def __init__(self, input_dim: int = 38, latent_dim: int = 8,
                 contamination: float = 0.1, random_state: int = 42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.contamination = contamination
        self.random_state = random_state
        self.is_fitted = False
        self._name = 'CNN Autoencoder'
        self.model = None
        self.threshold = None

        if not HAS_TF:
            raise ImportError("TensorFlow required for Conv1DAutoencoder")

        tf.random.set_seed(random_state)

    @property
    def name(self) -> str:
        return self._name

    def _build_model(self):
        """Build the autoencoder architecture."""
        # Reshape input to (input_dim, 1) for Conv1D
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Reshape((self.input_dim, 1))(inputs)

        # Encoder
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(16, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(8, 3, activation='relu', padding='same')(x)
        encoded = layers.GlobalAveragePooling1D()(x)
        encoded = layers.Dense(self.latent_dim, activation='relu')(encoded)

        # Decoder
        x = layers.Dense(self.input_dim // 4 * 8, activation='relu')(encoded)
        x = layers.Reshape((self.input_dim // 4, 8))(x)
        x = layers.Conv1DTranspose(16, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1DTranspose(32, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        # Final output to match input shape
        x = layers.Flatten()(x)
        decoded = layers.Dense(self.input_dim, activation='linear')(x)

        self.model = keras.Model(inputs, decoded)
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.1, verbose: int = 0) -> 'Conv1DAutoencoder':
        """Train the autoencoder on normal data."""
        self._build_model()

        # Normalize
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        self.history_ = self.model.fit(
            X_norm, X_norm,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor='val_loss'
                )
            ]
        )

        # Set threshold from training reconstruction errors
        train_errors = self._reconstruction_error(X_norm)
        self.threshold = np.percentile(train_errors, 100 * (1 - self.contamination))

        self.is_fitted = True
        logger.info(f"CNN Autoencoder trained. Threshold: {self.threshold:.4f}")
        return self

    def _reconstruction_error(self, X_norm: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction MSE."""
        reconstructed = self.model.predict(X_norm, verbose=0)
        mse = np.mean((X_norm - reconstructed) ** 2, axis=1)
        return mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies. -1 = anomaly, 1 = normal."""
        scores = self.score_samples(X)
        predictions = np.where(scores < -self.threshold, -1, 1)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (more negative = more anomalous)."""
        recalibrate = self._mean is None or self._std is None
        if recalibrate:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std
        errors = self._reconstruction_error(X_norm)
        if recalibrate:
            # Threshold from training doesn't match test-normalized errors;
            # recalibrate using contamination percentile on this data.
            self.threshold = np.percentile(errors, 100 * (1 - self.contamination))
            logger.info(f"Recalibrated CNN AE threshold: {self.threshold:.4f}")
        return -errors  # negate so more negative = more anomalous

    def save(self, filepath: str):
        """Save model weights and parameters."""
        self.model.save(filepath)
        np.savez(f"{filepath}_params.npz",
                 mean=self._mean, std=self._std, threshold=self.threshold)

    def load(self, filepath: str):
        """Load model weights and parameters."""
        self.model = keras.models.load_model(filepath, compile=False)
        params = np.load(f"{filepath}_params.npz")
        self._mean = params['mean']
        self._std = params['std']
        self.threshold = float(params['threshold'])
        self.is_fitted = True


class LSTMAutoencoder:
    """
    LSTM Autoencoder for sequence anomaly detection.

    Takes sequences of feature windows and detects anomalies
    based on reconstruction error of the sequence.

    Architecture:
        Encoder: LSTM(64) -> LSTM(32) -> latent
        Decoder: RepeatVector -> LSTM(32) -> LSTM(64) -> TimeDistributed(Dense)
    """

    def __init__(self, input_dim: int = 38, seq_length: int = 10,
                 latent_dim: int = 16, contamination: float = 0.1,
                 random_state: int = 42):
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.contamination = contamination
        self.random_state = random_state
        self.is_fitted = False
        self._name = 'LSTM Autoencoder'
        self.model = None
        self.threshold = None

        if not HAS_TF:
            raise ImportError("TensorFlow required for LSTMAutoencoder")

        tf.random.set_seed(random_state)

    @property
    def name(self) -> str:
        return self._name

    def _build_model(self):
        """Build LSTM autoencoder."""
        inputs = keras.Input(shape=(self.seq_length, self.input_dim))

        # Encoder
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.LSTM(self.latent_dim, return_sequences=False)(x)

        # Decoder
        x = layers.RepeatVector(self.seq_length)(x)
        x = layers.LSTM(self.latent_dim, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        decoded = layers.TimeDistributed(layers.Dense(self.input_dim))(x)

        self.model = keras.Model(inputs, decoded)
        self.model.compile(optimizer='adam', loss='mse')

    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert feature array to overlapping sequences."""
        n = len(X)
        if n < self.seq_length:
            # Pad with zeros
            padded = np.zeros((self.seq_length, X.shape[1]))
            padded[:n] = X
            return padded[np.newaxis, :]

        sequences = []
        for i in range(n - self.seq_length + 1):
            sequences.append(X[i:i + self.seq_length])
        return np.array(sequences)

    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32,
            validation_split: float = 0.1, verbose: int = 0) -> 'LSTMAutoencoder':
        """Train on feature windows."""
        self._build_model()

        # Normalize
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        # Create sequences
        sequences = self._create_sequences(X_norm)

        self.history_ = self.model.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True, monitor='val_loss'
                )
            ]
        )

        # Set threshold
        train_errors = self._sequence_errors(sequences)
        self.threshold = np.percentile(train_errors, 100 * (1 - self.contamination))

        self.is_fitted = True
        logger.info(f"LSTM Autoencoder trained. Threshold: {self.threshold:.4f}")
        return self

    def _sequence_errors(self, sequences: np.ndarray) -> np.ndarray:
        """Compute per-sequence reconstruction MSE."""
        reconstructed = self.model.predict(sequences, verbose=0)
        mse = np.mean((sequences - reconstructed) ** 2, axis=(1, 2))
        return mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies per window. Returns -1/1 array of len(X)."""
        scores = self.score_samples(X)
        predictions = np.where(scores < -self.threshold, -1, 1)
        return predictions

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return per-window anomaly scores."""
        recalibrate = self._mean is None or self._std is None
        if recalibrate:
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std
        sequences = self._create_sequences(X_norm)
        seq_errors = self._sequence_errors(sequences)

        # Map sequence errors back to per-window scores
        n = len(X)
        window_scores = np.zeros(n)
        window_counts = np.zeros(n)

        for i, err in enumerate(seq_errors):
            end = min(i + self.seq_length, n)
            window_scores[i:end] += err
            window_counts[i:end] += 1

        window_counts = np.maximum(window_counts, 1)
        per_window_errors = window_scores / window_counts
        if recalibrate:
            # Threshold from training doesn't match test-normalized errors;
            # recalibrate using contamination percentile on this data.
            self.threshold = np.percentile(per_window_errors, 100 * (1 - self.contamination))
            logger.info(f"Recalibrated LSTM AE threshold: {self.threshold:.4f}")
        return -per_window_errors  # negate: more negative = more anomalous

    def save(self, filepath: str):
        """Save model."""
        self.model.save(filepath)
        np.savez(f"{filepath}_params.npz",
                 mean=self._mean, std=self._std, threshold=self.threshold,
                 seq_length=self.seq_length)

    def load(self, filepath: str):
        """Load model."""
        self.model = keras.models.load_model(filepath, compile=False)
        params = np.load(f"{filepath}_params.npz")
        self._mean = params['mean']
        self._std = params['std']
        self.threshold = float(params['threshold'])
        self.seq_length = int(params['seq_length'])
        self.is_fitted = True
