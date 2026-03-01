"""
Hyperparameter tuning via grid search for anomaly detection models.
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from itertools import product
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import *
from backend.ml.preprocessing import LightCurvePreprocessor
from backend.ml.evaluation import AnomalyEvaluator
from backend.ml.model_registry import get_model, get_display_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Grid search for anomaly detection model hyperparameters."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.evaluator = AnomalyEvaluator(random_state=random_state)

    def grid_search(self, X: np.ndarray, y: np.ndarray,
                    model_name: str, param_grid: Dict[str, List],
                    n_folds: int = N_CV_FOLDS,
                    metric: str = 'f1') -> Dict:
        """
        Grid search over hyperparameters using cross-validation.

        Args:
            X: Feature array
            y: Labels
            model_name: Model identifier
            param_grid: Dict of param_name -> list of values
            n_folds: Number of CV folds
            metric: Metric to optimize ('f1', 'precision', 'recall', 'roc_auc')

        Returns:
            Dict with best params, all results, and ranking
        """
        logger.info(f"Grid search for {model_name} over {param_grid}")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(product(*param_values))

        results = []

        for combo in all_combos:
            params = dict(zip(param_names, combo))
            logger.info(f"  Testing: {params}")

            def factory(p=params):
                return get_model(model_name, random_state=self.random_state, **p)

            try:
                cv_result = self.evaluator.cross_validate(
                    X, y, factory, n_folds=n_folds, strategy=CV_STRATEGY
                )
                metric_key = f'{metric}_mean'
                score = cv_result['aggregated'].get(metric_key, float('nan'))

                results.append({
                    'params': params,
                    'score': score,
                    'all_metrics': cv_result['aggregated'],
                })
                logger.info(f"    {metric}={score:.4f}")

            except Exception as e:
                logger.warning(f"    Failed: {e}")
                results.append({
                    'params': params,
                    'score': float('nan'),
                    'error': str(e),
                })

        # Sort by score
        valid_results = [r for r in results if not np.isnan(r['score'])]
        valid_results.sort(key=lambda x: x['score'], reverse=True)

        best = valid_results[0] if valid_results else None

        output = {
            'model': model_name,
            'metric': metric,
            'best_params': best['params'] if best else None,
            'best_score': best['score'] if best else float('nan'),
            'all_results': results,
            'n_combinations': len(all_combos),
        }

        logger.info(f"Best: {output['best_params']} -> {metric}={output['best_score']:.4f}")
        return output

    def tune_isolation_forest(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune Isolation Forest hyperparameters."""
        grid = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'n_estimators': [50, 100, 200],
        }
        return self.grid_search(X, y, 'isolation_forest', grid)

    def tune_lof(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune LOF hyperparameters."""
        grid = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'n_neighbors': [10, 20, 30, 50],
        }
        return self.grid_search(X, y, 'lof', grid)

    def tune_ocsvm(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune One-Class SVM hyperparameters."""
        grid = {
            'contamination': [0.05, 0.1, 0.15, 0.2],
            'kernel': ['rbf', 'poly'],
        }
        return self.grid_search(X, y, 'ocsvm', grid)

    def tune_all(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Tune all classical models."""
        results = {}
        results['isolation_forest'] = self.tune_isolation_forest(X, y)
        results['lof'] = self.tune_lof(X, y)
        results['ocsvm'] = self.tune_ocsvm(X, y)

        # Save
        output_path = Path(RAW_DIR) / 'hyperparameter_tuning.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results


if __name__ == '__main__':
    from experiments.run_experiments import ExperimentRunner

    runner = ExperimentRunner()
    X, y = runner.load_dataset()

    tuner = HyperparameterTuner()
    results = tuner.tune_all(X, y)

    print("\n=== Best hyperparameters ===")
    for model, result in results.items():
        print(f"{model}: {result['best_params']} -> F1={result['best_score']:.4f}")
