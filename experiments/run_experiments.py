"""
Experiment runner for producing all paper figures and tables.

7 experiments:
1. Model Comparison Table — all models x 5-fold CV
2. Feature Ablation — cumulative feature groups
3. Per-Type Detection — transit/flare/outlier/normal detection rates
4. Hyperparameter Sensitivity — contamination, window_size sweeps
5. Detection Examples — qualitative figure with example light curves
6. Feature Importance — from best tree-based model
7. Statistical Significance — paired tests between top models
"""

import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.config import *
from backend.ml.preprocessing import LightCurvePreprocessor
from backend.ml.evaluation import AnomalyEvaluator
from backend.ml.figures import PaperFigureGenerator
from backend.ml.model_registry import get_model, list_models, get_display_name
from backend.ml.feature_names import get_feature_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Runs all experiments and generates paper figures/tables.
    """

    def __init__(self, data_dir: str = 'data/labeled',
                 results_dir: str = RESULTS_DIR):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.figures_dir = Path(FIGURES_DIR)
        self.tables_dir = Path(TABLES_DIR)
        self.raw_dir = Path(RAW_DIR)

        for d in [self.results_dir, self.figures_dir, self.tables_dir, self.raw_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.preprocessor = LightCurvePreprocessor()
        self.evaluator = AnomalyEvaluator(random_state=RANDOM_SEED)
        self.figure_gen = PaperFigureGenerator(output_dir=str(self.figures_dir))

        self._features_cache = {}
        self._labels_cache = {}

    def load_dataset(self, feature_groups: Optional[List[str]] = None,
                     window_size: int = DEFAULT_WINDOW_SIZE):
        """Load labeled dataset and extract features."""
        cache_key = (tuple(feature_groups or DEFAULT_FEATURE_GROUPS), window_size)
        if cache_key in self._features_cache:
            return self._features_cache[cache_key], self._labels_cache[cache_key]

        metadata_path = self.data_dir / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No dataset at {self.data_dir}. Run scripts/download_dataset.py first."
            )

        metadata = pd.read_csv(metadata_path)
        all_features = []
        all_labels = []

        for _, row in metadata.iterrows():
            lc_path = self.data_dir / 'lightcurves' / row['filename']
            if not lc_path.exists():
                continue

            df = pd.read_csv(lc_path)
            if 'label' not in df.columns:
                continue

            # Preprocess (don't normalize labels column)
            labels = df['label'].values
            df_proc = self.preprocessor.preprocess(df, normalize=True)

            # Extract features
            groups = feature_groups or DEFAULT_FEATURE_GROUPS
            features = self.preprocessor.extract_features(df_proc, window_size, groups)

            # Map labels to windows (any transit point makes window anomalous)
            stride = max(1, window_size // 4)
            window_labels = []
            for i in range(0, len(df_proc) - window_size + 1, stride):
                window_lab = labels[i:i + window_size]
                window_labels.append(int(window_lab.max() > 0))

            n_windows = min(len(features), len(window_labels))
            all_features.append(features[:n_windows])
            all_labels.append(np.array(window_labels[:n_windows]))

        if not all_features:
            raise ValueError("No valid labeled data found")

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        self._features_cache[cache_key] = X
        self._labels_cache[cache_key] = y

        logger.info(f"Loaded dataset: {X.shape[0]} windows, {X.shape[1]} features, "
                     f"{y.sum()} anomalies ({y.mean()*100:.1f}%)")
        return X, y

    def experiment_1_model_comparison(self, models: Optional[List[str]] = None):
        """
        Experiment 1: Model Comparison Table.
        All models x 5-fold stratified CV -> precision, recall, F1, ROC-AUC, PR-AUC.
        """
        logger.info("=== Experiment 1: Model Comparison ===")
        models = models or CLASSICAL_MODELS
        X, y = self.load_dataset()

        results = {}
        for model_name in models:
            logger.info(f"  Evaluating: {model_name}")
            display_name = get_display_name(model_name)

            def factory():
                return get_model(model_name, contamination=DEFAULT_CONTAMINATION,
                                 random_state=RANDOM_SEED)

            cv_result = self.evaluator.cross_validate(
                X, y, factory, n_folds=N_CV_FOLDS, strategy=CV_STRATEGY
            )
            results[display_name] = cv_result['aggregated']

            logger.info(f"    F1={cv_result['aggregated']['f1_mean']:.3f} "
                        f"+/- {cv_result['aggregated']['f1_std']:.3f}")

        # Generate LaTeX table
        self.figure_gen.generate_latex_table(
            results, filename='model_comparison.tex',
            caption='Model comparison with 5-fold stratified cross-validation.'
        )

        # Save raw results
        self._save_json(results, 'experiment_1_model_comparison.json')
        return results

    def experiment_2_feature_ablation(self, model_name: str = 'ensemble'):
        """
        Experiment 2: Feature Ablation.
        Best model with cumulative feature groups.
        """
        logger.info("=== Experiment 2: Feature Ablation ===")
        ablation_results = {}

        for groups, label in zip(ABLATION_ORDER, ABLATION_LABELS):
            logger.info(f"  Feature groups: {groups}")
            X, y = self.load_dataset(feature_groups=groups)

            def factory():
                return get_model(model_name, contamination=DEFAULT_CONTAMINATION,
                                 random_state=RANDOM_SEED)

            cv_result = self.evaluator.cross_validate(
                X, y, factory, n_folds=N_CV_FOLDS, strategy=CV_STRATEGY
            )
            ablation_results[label] = cv_result['aggregated']['f1_mean']
            logger.info(f"    {label}: F1={cv_result['aggregated']['f1_mean']:.3f}")

        # Plot
        self.figure_gen.plot_ablation_results(
            ablation_results, metric_name='F1 Score',
            filename='feature_ablation.pdf'
        )

        self._save_json(ablation_results, 'experiment_2_feature_ablation.json')
        return ablation_results

    def experiment_3_per_type_detection(self, model_name: str = 'ensemble'):
        """
        Experiment 3: Per-Type Detection Rates.
        Metrics broken down by anomaly type (transit/flare/outlier/normal).
        """
        logger.info("=== Experiment 3: Per-Type Detection ===")
        X, y = self.load_dataset()

        # Load anomaly types from metadata
        metadata = pd.read_csv(self.data_dir / 'metadata.csv')

        # Build anomaly type array aligned with windows
        # For now, use label_type from metadata
        anomaly_types = np.where(y == 1, 'anomaly', 'normal')

        # Train on full data (or best fold)
        model = get_model(model_name, contamination=DEFAULT_CONTAMINATION,
                          random_state=RANDOM_SEED)
        model.fit(X)
        y_pred_raw = model.predict(X)
        y_pred = (y_pred_raw == -1).astype(int) if y_pred_raw.min() < 0 else y_pred_raw

        scores = None
        if hasattr(model, 'score_samples'):
            scores = -model.score_samples(X)

        per_type = self.evaluator.compute_per_type_metrics(y, y_pred, anomaly_types, scores)

        self._save_json(
            {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                 for kk, vv in v.items()}
             for k, v in per_type.items()},
            'experiment_3_per_type_detection.json'
        )
        return per_type

    def experiment_4_hyperparameter_sensitivity(self, model_name: str = 'isolation_forest'):
        """
        Experiment 4: Hyperparameter Sensitivity.
        Sweep contamination and window_size.
        """
        logger.info("=== Experiment 4: Hyperparameter Sensitivity ===")
        results = {}

        # Contamination sweep
        logger.info("  Sweeping contamination...")
        contamination_results = {}
        X, y = self.load_dataset()

        for c in HYPERPARAM_GRIDS['contamination']:
            def factory(c=c):
                return get_model(model_name, contamination=c, random_state=RANDOM_SEED)

            cv_result = self.evaluator.cross_validate(
                X, y, factory, n_folds=N_CV_FOLDS, strategy=CV_STRATEGY
            )
            contamination_results[str(c)] = cv_result['aggregated']['f1_mean']
            logger.info(f"    contamination={c}: F1={cv_result['aggregated']['f1_mean']:.3f}")

        results['contamination'] = contamination_results

        # Window size sweep
        logger.info("  Sweeping window_size...")
        window_results = {}
        for ws in HYPERPARAM_GRIDS['window_size']:
            try:
                X_ws, y_ws = self.load_dataset(window_size=ws)

                def factory():
                    return get_model(model_name, contamination=DEFAULT_CONTAMINATION,
                                     random_state=RANDOM_SEED)

                cv_result = self.evaluator.cross_validate(
                    X_ws, y_ws, factory, n_folds=N_CV_FOLDS, strategy=CV_STRATEGY
                )
                window_results[str(ws)] = cv_result['aggregated']['f1_mean']
                logger.info(f"    window_size={ws}: F1={cv_result['aggregated']['f1_mean']:.3f}")
            except Exception as e:
                logger.warning(f"    window_size={ws} failed: {e}")

        results['window_size'] = window_results

        self._save_json(results, 'experiment_4_hyperparameter_sensitivity.json')
        return results

    def experiment_5_detection_examples(self, model_name: str = 'ensemble', n_examples: int = 6):
        """
        Experiment 5: Detection Example Figures.
        Show example light curves with true labels and predictions.
        """
        logger.info("=== Experiment 5: Detection Examples ===")

        metadata = pd.read_csv(self.data_dir / 'metadata.csv')
        model = get_model(model_name, contamination=DEFAULT_CONTAMINATION,
                          random_state=RANDOM_SEED)

        # Collect training data from all light curves
        all_features = []
        for _, row in metadata.iterrows():
            lc_path = self.data_dir / 'lightcurves' / row['filename']
            if not lc_path.exists():
                continue
            df = pd.read_csv(lc_path)
            df_proc = self.preprocessor.preprocess(df, normalize=True)
            features = self.preprocessor.extract_features(df_proc, DEFAULT_WINDOW_SIZE)
            all_features.append(features)

        if all_features:
            X_all = np.vstack(all_features)
            model.fit(X_all)

        # Select examples
        examples = []
        transit_rows = metadata[metadata['label_type'] == 'transit'].head(3)
        normal_rows = metadata[metadata['label_type'] == 'normal'].head(3)
        selected = pd.concat([transit_rows, normal_rows]).head(n_examples)

        from backend.ml.models import map_window_predictions_to_points

        for _, row in selected.iterrows():
            lc_path = self.data_dir / 'lightcurves' / row['filename']
            if not lc_path.exists():
                continue

            df = pd.read_csv(lc_path)
            true_labels = df['label'].values if 'label' in df.columns else None

            df_proc = self.preprocessor.preprocess(df, normalize=True)
            features = self.preprocessor.extract_features(df_proc, DEFAULT_WINDOW_SIZE)

            predictions = model.predict(features)
            scores = model.score_samples(features) if hasattr(model, 'score_samples') else np.zeros(len(features))

            stride = max(1, DEFAULT_WINDOW_SIZE // 4)
            pred_mask, _ = map_window_predictions_to_points(
                predictions, scores, len(df_proc), DEFAULT_WINDOW_SIZE, stride, method='vote'
            )

            examples.append({
                'time': df_proc['time'].values,
                'flux': df_proc['flux'].values,
                'true_labels': true_labels[:len(df_proc)] if true_labels is not None else None,
                'pred_labels': pred_mask.astype(int),
                'title': f"{row['target_id']} ({row['label_type']})",
            })

        if examples:
            self.figure_gen.plot_detection_examples(examples, filename='detection_examples.pdf')

        return examples

    def experiment_6_feature_importance(self, model_name: str = 'isolation_forest'):
        """
        Experiment 6: Feature Importance.
        Extract from best tree-based model (Isolation Forest).
        """
        logger.info("=== Experiment 6: Feature Importance ===")
        X, y = self.load_dataset()

        # Train IF and extract feature importances from trees
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iforest = IsolationForest(
            contamination=DEFAULT_CONTAMINATION,
            n_estimators=200,
            random_state=RANDOM_SEED,
        )
        iforest.fit(X_scaled)

        # Compute feature importance via mean decrease in path length
        # Approximate: use feature importances from each tree
        importances = np.zeros(X.shape[1])
        for tree in iforest.estimators_:
            tree_importances = tree.feature_importances_
            importances += tree_importances
        importances /= len(iforest.estimators_)

        feature_names = get_feature_names()
        # Pad or truncate names to match feature count
        while len(feature_names) < len(importances):
            feature_names.append(f'feature_{len(feature_names)}')
        feature_names = feature_names[:len(importances)]

        self.figure_gen.plot_feature_importance(
            importances, feature_names, top_k=min(20, len(importances)),
            filename='feature_importance.pdf'
        )

        result = {name: float(imp) for name, imp in zip(feature_names, importances)}
        self._save_json(result, 'experiment_6_feature_importance.json')
        return result

    def experiment_7_statistical_significance(self):
        """
        Experiment 7: Statistical Significance.
        Paired tests between top models from Experiment 1.
        """
        logger.info("=== Experiment 7: Statistical Significance ===")
        X, y = self.load_dataset()

        # Get per-fold F1 scores for each model
        model_fold_f1s = {}
        for model_name in CLASSICAL_MODELS:
            display_name = get_display_name(model_name)

            def factory(name=model_name):
                return get_model(name, contamination=DEFAULT_CONTAMINATION,
                                 random_state=RANDOM_SEED)

            cv_result = self.evaluator.cross_validate(
                X, y, factory, n_folds=N_CV_FOLDS, strategy=CV_STRATEGY
            )
            model_fold_f1s[display_name] = [
                fold['f1'] for fold in cv_result['fold_metrics']
            ]

        # Pairwise tests
        model_names = list(model_fold_f1s.keys())
        pairwise_results = {}

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a = model_names[i]
                name_b = model_names[j]
                test = self.evaluator.paired_test(
                    model_fold_f1s[name_a], model_fold_f1s[name_b]
                )
                key = f"{name_a} vs {name_b}"
                pairwise_results[key] = test
                logger.info(f"  {key}: diff={test['mean_diff']:.4f}, "
                            f"p={test['ttest_pvalue']:.4f}")

        self._save_json(pairwise_results, 'experiment_7_statistical_significance.json')
        return pairwise_results

    def run_all(self, models: Optional[List[str]] = None, skip_deep: bool = True):
        """Run all 7 experiments."""
        logger.info("Running all experiments...")

        models = models or CLASSICAL_MODELS
        if not skip_deep:
            models = models + DEEP_MODELS

        results = {}
        results['exp1'] = self.experiment_1_model_comparison(models)
        results['exp2'] = self.experiment_2_feature_ablation()
        results['exp3'] = self.experiment_3_per_type_detection()
        results['exp4'] = self.experiment_4_hyperparameter_sensitivity()
        results['exp5'] = self.experiment_5_detection_examples()
        results['exp6'] = self.experiment_6_feature_importance()
        results['exp7'] = self.experiment_7_statistical_significance()

        logger.info("All experiments complete!")
        return results

    def _save_json(self, data, filename):
        """Save results as JSON."""
        filepath = self.raw_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved {filepath}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run paper experiments')
    parser.add_argument('--experiment', type=int, default=0,
                        help='Specific experiment number (0=all)')
    parser.add_argument('--data-dir', type=str, default='data/labeled')
    parser.add_argument('--skip-deep', action='store_true', default=True)

    args = parser.parse_args()

    runner = ExperimentRunner(data_dir=args.data_dir)

    if args.experiment == 0:
        runner.run_all(skip_deep=args.skip_deep)
    elif args.experiment == 1:
        runner.experiment_1_model_comparison()
    elif args.experiment == 2:
        runner.experiment_2_feature_ablation()
    elif args.experiment == 3:
        runner.experiment_3_per_type_detection()
    elif args.experiment == 4:
        runner.experiment_4_hyperparameter_sensitivity()
    elif args.experiment == 5:
        runner.experiment_5_detection_examples()
    elif args.experiment == 6:
        runner.experiment_6_feature_importance()
    elif args.experiment == 7:
        runner.experiment_7_statistical_significance()
