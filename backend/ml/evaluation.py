"""
Evaluation framework for anomaly detection models.

Provides proper ML evaluation with precision/recall/F1/ROC-AUC,
cross-validation, confidence intervals, and statistical tests.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, KFold
from typing import Dict, List, Optional, Callable, Tuple
import logging
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """
    Comprehensive evaluation for anomaly detection models.
    Supports binary classification metrics, cross-validation,
    bootstrap confidence intervals, and paired statistical tests.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                        y_scores: Optional[np.ndarray] = None) -> Dict:
        """
        Compute all standard classification metrics.

        Args:
            y_true: Ground truth binary labels (0=normal, 1=anomaly)
            y_pred: Predicted binary labels (0=normal, 1=anomaly)
            y_scores: Anomaly scores (higher = more anomalous). If scores
                      from sklearn (more negative = more anomalous), negate them first.

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'n_true_anomalies': int(y_true.sum()),
            'n_pred_anomalies': int(y_pred.sum()),
            'n_total': len(y_true),
            'anomaly_rate_true': float(y_true.mean()),
            'anomaly_rate_pred': float(y_pred.mean()),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])

        # Score-based metrics (require continuous scores)
        if y_scores is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
            except ValueError:
                metrics['roc_auc'] = float('nan')
            try:
                metrics['pr_auc'] = float(average_precision_score(y_true, y_scores))
            except ValueError:
                metrics['pr_auc'] = float('nan')
        else:
            metrics['roc_auc'] = float('nan')
            metrics['pr_auc'] = float('nan')

        return metrics

    def compute_per_type_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  anomaly_types: np.ndarray,
                                  y_scores: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """
        Compute metrics broken down by anomaly type.

        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly)
            y_pred: Predicted labels
            anomaly_types: Array of strings indicating type for each point
                           (e.g., 'normal', 'transit', 'flare', 'outlier')
            y_scores: Optional anomaly scores

        Returns:
            Dict mapping type name to metrics dict
        """
        results = {}
        unique_types = np.unique(anomaly_types)

        for atype in unique_types:
            mask = anomaly_types == atype
            if mask.sum() == 0:
                continue

            type_scores = y_scores[mask] if y_scores is not None else None
            results[atype] = self.compute_metrics(
                y_true[mask], y_pred[mask], type_scores
            )
            results[atype]['n_points'] = int(mask.sum())

        return results

    def cross_validate(self, features: np.ndarray, labels: np.ndarray,
                       model_factory: Callable, n_folds: int = 5,
                       strategy: str = 'stratified') -> Dict:
        """
        K-fold cross-validation for anomaly detection.

        Args:
            features: Feature array (n_samples, n_features)
            labels: Binary labels (n_samples,)
            model_factory: Callable that returns a fresh model instance with fit()/predict()/score_samples()
            n_folds: Number of CV folds
            strategy: 'stratified' or 'temporal'
                - stratified: preserves anomaly proportion in each fold
                - temporal: sequential splits (no shuffling)

        Returns:
            Dict with per-fold and aggregated metrics
        """
        if strategy == 'stratified':
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                     random_state=self.random_state)
            splits = kfold.split(features, labels)
        elif strategy == 'temporal':
            kfold = KFold(n_splits=n_folds, shuffle=False)
            splits = kfold.split(features)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Train model
            model = model_factory()
            model.fit(X_train)

            # Predict
            y_pred_raw = model.predict(X_test)
            # Convert sklearn convention (-1/1) to (1/0)
            y_pred = (y_pred_raw == -1).astype(int) if y_pred_raw.min() < 0 else y_pred_raw

            # Get scores
            y_scores = None
            if hasattr(model, 'score_samples'):
                try:
                    raw_scores = model.score_samples(X_test)
                    y_scores = -raw_scores  # negate: higher = more anomalous
                except Exception:
                    pass

            fold_result = self.compute_metrics(y_test, y_pred, y_scores)
            fold_result['fold'] = fold_idx
            fold_metrics.append(fold_result)

            logger.info(
                f"Fold {fold_idx}: P={fold_result['precision']:.3f} "
                f"R={fold_result['recall']:.3f} F1={fold_result['f1']:.3f}"
            )

        # Aggregate across folds
        metric_names = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        aggregated = {}
        for m in metric_names:
            values = [f[m] for f in fold_metrics if not np.isnan(f[m])]
            if values:
                aggregated[f'{m}_mean'] = float(np.mean(values))
                aggregated[f'{m}_std'] = float(np.std(values))
            else:
                aggregated[f'{m}_mean'] = float('nan')
                aggregated[f'{m}_std'] = float('nan')

        return {
            'fold_metrics': fold_metrics,
            'aggregated': aggregated,
            'n_folds': n_folds,
            'strategy': strategy,
        }

    def bootstrap_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        y_scores: Optional[np.ndarray] = None,
                                        n_bootstrap: int = 1000,
                                        confidence: float = 0.95) -> Dict:
        """
        Compute bootstrap confidence intervals for all metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_scores: Anomaly scores
            n_bootstrap: Number of bootstrap iterations
            confidence: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Dict mapping metric name to (mean, lower, upper)
        """
        rng = np.random.RandomState(self.random_state)
        n = len(y_true)

        bootstrap_metrics = {
            'precision': [], 'recall': [], 'f1': [],
            'roc_auc': [], 'pr_auc': [],
        }

        for _ in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            bt_true = y_true[idx]
            bt_pred = y_pred[idx]
            bt_scores = y_scores[idx] if y_scores is not None else None

            # Skip degenerate bootstrap samples
            if len(np.unique(bt_true)) < 2:
                continue

            metrics = self.compute_metrics(bt_true, bt_pred, bt_scores)
            for key in bootstrap_metrics:
                if key in metrics and not np.isnan(metrics[key]):
                    bootstrap_metrics[key].append(metrics[key])

        alpha = (1 - confidence) / 2
        results = {}
        for key, values in bootstrap_metrics.items():
            if len(values) > 10:
                values = np.array(values)
                results[key] = {
                    'mean': float(np.mean(values)),
                    'lower': float(np.percentile(values, 100 * alpha)),
                    'upper': float(np.percentile(values, 100 * (1 - alpha))),
                    'std': float(np.std(values)),
                }
            else:
                results[key] = {
                    'mean': float('nan'), 'lower': float('nan'),
                    'upper': float('nan'), 'std': float('nan'),
                }

        return results

    def paired_test(self, metrics_a: List[float], metrics_b: List[float]) -> Dict:
        """
        Paired statistical tests between two models (e.g., from CV folds).

        Args:
            metrics_a: List of metric values from model A (one per fold)
            metrics_b: List of metric values from model B (one per fold)

        Returns:
            Dict with t-test and Wilcoxon test results
        """
        a = np.array(metrics_a)
        b = np.array(metrics_b)

        results = {
            'mean_a': float(np.mean(a)),
            'mean_b': float(np.mean(b)),
            'mean_diff': float(np.mean(a - b)),
        }

        # Paired t-test
        if len(a) >= 3:
            t_stat, t_pvalue = sp_stats.ttest_rel(a, b)
            results['ttest_statistic'] = float(t_stat)
            results['ttest_pvalue'] = float(t_pvalue)
        else:
            results['ttest_statistic'] = float('nan')
            results['ttest_pvalue'] = float('nan')

        # Wilcoxon signed-rank test
        if len(a) >= 6:  # needs enough samples
            try:
                w_stat, w_pvalue = sp_stats.wilcoxon(a, b)
                results['wilcoxon_statistic'] = float(w_stat)
                results['wilcoxon_pvalue'] = float(w_pvalue)
            except ValueError:
                results['wilcoxon_statistic'] = float('nan')
                results['wilcoxon_pvalue'] = float('nan')
        else:
            results['wilcoxon_statistic'] = float('nan')
            results['wilcoxon_pvalue'] = float('nan')

        return results

    def get_roc_curve_data(self, y_true: np.ndarray,
                           y_scores: np.ndarray) -> Dict:
        """Get ROC curve data for plotting."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc}

    def get_pr_curve_data(self, y_true: np.ndarray,
                          y_scores: np.ndarray) -> Dict:
        """Get Precision-Recall curve data for plotting."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        return {'precision': precision, 'recall': recall,
                'thresholds': thresholds, 'ap': ap}
