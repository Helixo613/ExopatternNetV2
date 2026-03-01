"""
Publication-quality figure generation for anomaly detection results.

Generates matplotlib figures suitable for conference/journal papers:
ROC curves, PR curves, confusion matrices, feature importance,
detection examples, ablation results, and LaTeX tables.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Use non-interactive backend for server/script usage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# Paper-quality defaults
PAPER_RC = {
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
}


class PaperFigureGenerator:
    """
    Generates publication-quality figures for the anomaly detection paper.
    """

    def __init__(self, output_dir: str = 'results/figures', style: str = 'paper'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if style == 'paper':
            plt.rcParams.update(PAPER_RC)

        # Color palette for models
        self.model_colors = {
            'Isolation Forest': '#1f77b4',
            'LOF': '#ff7f0e',
            'One-Class SVM': '#2ca02c',
            'DBSCAN': '#d62728',
            'Ensemble (IF+LOF)': '#9467bd',
            'BLS': '#8c564b',
            'CNN Autoencoder': '#e377c2',
            'LSTM Autoencoder': '#7f7f7f',
        }

    def _get_color(self, name: str) -> str:
        """Get color for a model name, with fallback."""
        if name in self.model_colors:
            return self.model_colors[name]
        # Generate deterministic color from name hash
        h = hash(name) % 360
        return f'C{abs(hash(name)) % 10}'

    def plot_roc_curves(self, results: Dict[str, Dict], filename: str = 'roc_curves.pdf'):
        """
        Plot multi-model ROC curves on a single figure.

        Args:
            results: Dict mapping model_name -> {'fpr': array, 'tpr': array, 'auc': float}
            filename: Output filename
        """
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        for name, data in results.items():
            color = self._get_color(name)
            ax.plot(data['fpr'], data['tpr'],
                    label=f"{name} (AUC={data['auc']:.3f})",
                    color=color, linewidth=1.5)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved ROC curves to {filepath}")

    def plot_pr_curves(self, results: Dict[str, Dict], filename: str = 'pr_curves.pdf'):
        """
        Plot multi-model Precision-Recall curves.

        Args:
            results: Dict mapping model_name -> {'precision': array, 'recall': array, 'ap': float}
        """
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        for name, data in results.items():
            color = self._get_color(name)
            ax.plot(data['recall'], data['precision'],
                    label=f"{name} (AP={data['ap']:.3f})",
                    color=color, linewidth=1.5)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)

        filepath = self.output_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved PR curves to {filepath}")

    def plot_confusion_matrices(self, results: Dict[str, Dict],
                                 filename: str = 'confusion_matrices.pdf'):
        """
        Plot side-by-side confusion matrices for multiple models.

        Args:
            results: Dict mapping model_name -> {'tp', 'fp', 'fn', 'tn'}
        """
        n_models = len(results)
        cols = min(4, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3 * rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (name, data) in enumerate(results.items()):
            ax = axes[idx]
            cm = np.array([[data['tn'], data['fp']], [data['fn'], data['tp']]])

            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_title(name, fontsize=9)

            # Annotate cells
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=10, color='white' if cm[i, j] > cm.max() / 2 else 'black')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'Anomaly'], fontsize=8)
            ax.set_yticklabels(['Normal', 'Anomaly'], fontsize=8)
            ax.set_ylabel('True')
            ax.set_xlabel('Predicted')

        # Hide empty axes
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved confusion matrices to {filepath}")

    def plot_feature_importance(self, importances: np.ndarray,
                                 feature_names: List[str],
                                 top_k: int = 20,
                                 filename: str = 'feature_importance.pdf'):
        """
        Plot horizontal bar chart of feature importances.

        Args:
            importances: Feature importance scores
            feature_names: Feature names
            top_k: Show top K features
        """
        # Sort by importance
        idx = np.argsort(importances)[::-1][:top_k]

        fig, ax = plt.subplots(1, 1, figsize=(5, 0.3 * top_k + 1))

        names = [feature_names[i] for i in idx]
        values = importances[idx]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, align='center', color='#1f77b4', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_k} Feature Importances')
        ax.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved feature importance to {filepath}")

    def plot_detection_examples(self, examples: List[Dict],
                                 filename: str = 'detection_examples.pdf'):
        """
        Plot example light curves with true labels and predictions.

        Args:
            examples: List of dicts with keys:
                'time', 'flux', 'true_labels', 'pred_labels', 'title'
        """
        n = len(examples)
        fig, axes = plt.subplots(n, 1, figsize=(8, 2.5 * n), sharex=False)
        if n == 1:
            axes = [axes]

        for idx, ex in enumerate(examples):
            ax = axes[idx]
            time = ex['time']
            flux = ex['flux']
            true_labels = ex.get('true_labels')
            pred_labels = ex.get('pred_labels')

            # Plot normal points
            normal_mask = np.ones(len(time), dtype=bool)
            if true_labels is not None:
                normal_mask &= (true_labels == 0)

            ax.scatter(time[normal_mask], flux[normal_mask],
                       s=1, c='#1f77b4', alpha=0.4, label='Normal', rasterized=True)

            # True anomalies
            if true_labels is not None:
                true_mask = true_labels == 1
                ax.scatter(time[true_mask], flux[true_mask],
                           s=8, c='red', marker='v', alpha=0.7, label='True anomaly')

            # Predicted anomalies (if different from true)
            if pred_labels is not None:
                pred_mask = pred_labels == 1
                # Only show false positives and false negatives distinctly
                if true_labels is not None:
                    fp_mask = pred_mask & (true_labels == 0)
                    ax.scatter(time[fp_mask], flux[fp_mask],
                               s=12, c='orange', marker='x', alpha=0.7, label='False positive')
                else:
                    ax.scatter(time[pred_mask], flux[pred_mask],
                               s=8, c='red', marker='v', alpha=0.7, label='Predicted anomaly')

            ax.set_ylabel('Flux')
            ax.set_title(ex.get('title', f'Example {idx + 1}'), fontsize=10)
            ax.legend(loc='upper right', fontsize=7, markerscale=1.5)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel('Time (days)')
        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved detection examples to {filepath}")

    def plot_ablation_results(self, results: Dict[str, float],
                               metric_name: str = 'F1 Score',
                               filename: str = 'ablation_results.pdf'):
        """
        Plot bar chart of metric by feature group (ablation study).

        Args:
            results: Dict mapping feature_group_label -> metric_value
                     e.g., {'Statistical': 0.72, '+Frequency': 0.78, ...}
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        labels = list(results.keys())
        values = list(results.values())
        x = np.arange(len(labels))

        bars = ax.bar(x, values, color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)

        # Color the best bar differently
        best_idx = np.argmax(values)
        bars[best_idx].set_color('#2ca02c')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel(metric_name)
        ax.set_title(f'Feature Ablation Study ({metric_name})')
        ax.grid(True, axis='y', alpha=0.3)

        # Annotate bars with values
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath)
        plt.close(fig)
        logger.info(f"Saved ablation results to {filepath}")

    def generate_latex_table(self, results: Dict[str, Dict],
                              metrics: List[str] = None,
                              filename: str = 'model_comparison.tex',
                              caption: str = 'Model comparison results.') -> str:
        """
        Generate a LaTeX-formatted comparison table.

        Args:
            results: Dict mapping model_name -> metrics dict
                     Each metrics dict should have keys like 'precision_mean', 'precision_std', etc.
            metrics: List of metric names to include (default: precision, recall, f1, roc_auc)
            filename: Output .tex filename
            caption: Table caption

        Returns:
            LaTeX table string
        """
        if metrics is None:
            metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

        # Build table
        header_map = {
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'roc_auc': 'ROC-AUC',
            'pr_auc': 'PR-AUC',
        }

        col_spec = 'l' + 'c' * len(metrics)
        headers = ['Model'] + [header_map.get(m, m) for m in metrics]

        lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            f'\\caption{{{caption}}}',
            f'\\begin{{tabular}}{{{col_spec}}}',
            '\\toprule',
            ' & '.join(headers) + ' \\\\',
            '\\midrule',
        ]

        # Find best values for bolding
        best_vals = {}
        for m in metrics:
            vals = []
            for model_metrics in results.values():
                key = f'{m}_mean'
                if key in model_metrics and not np.isnan(model_metrics[key]):
                    vals.append(model_metrics[key])
            best_vals[m] = max(vals) if vals else None

        for model_name, model_metrics in results.items():
            row = [model_name.replace('_', ' ')]
            for m in metrics:
                mean_key = f'{m}_mean'
                std_key = f'{m}_std'
                mean_val = model_metrics.get(mean_key, float('nan'))
                std_val = model_metrics.get(std_key, float('nan'))

                if np.isnan(mean_val):
                    cell = '--'
                elif np.isnan(std_val):
                    cell = f'{mean_val:.3f}'
                else:
                    cell = f'{mean_val:.3f} $\\pm$ {std_val:.3f}'

                # Bold the best
                if best_vals[m] is not None and not np.isnan(mean_val):
                    if abs(mean_val - best_vals[m]) < 1e-6:
                        cell = f'\\textbf{{{cell}}}'

                row.append(cell)

            lines.append(' & '.join(row) + ' \\\\')

        lines.extend([
            '\\bottomrule',
            '\\end{tabular}',
            '\\end{table}',
        ])

        latex = '\n'.join(lines)

        filepath = self.output_dir.parent / 'tables' / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(latex)

        logger.info(f"Saved LaTeX table to {filepath}")
        return latex
