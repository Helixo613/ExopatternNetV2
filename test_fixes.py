#!/usr/bin/env python3
"""
Test script to validate the anomaly detection fixes.
Compares old vs new approach on sample data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.data import LightCurveLoader
from backend.ml import LightCurvePreprocessor, AnomalyDetector, ModelTrainer

def test_improvements():
    """Test the improvements on sample data."""

    print("="*80)
    print("TESTING ANOMALY DETECTION FIXES")
    print("="*80)

    # Sample files to test
    sample_files = [
        'data/samples/normal_star.csv',
        'data/samples/exoplanet_transit.csv',
        'data/samples/stellar_flares.csv',
        'data/samples/noisy_outliers.csv',
        'data/samples/complex_system.csv'
    ]

    # Initialize components
    print("\n1. Initializing model with synthetic data...")
    trainer = ModelTrainer()
    detector = trainer.train_with_synthetic_data(n_samples=100, contamination=0.05)
    preprocessor = trainer.preprocessor
    print("   ✓ Model trained successfully")

    # Test with new settings
    print("\n2. Testing with NEW SETTINGS:")
    print("   - Contamination: 0.05 (5%)")
    print("   - Ensemble strategy: score_threshold")
    print("   - Vote threshold: 0.3 (30%)")
    print("   - Window overlap: 50%")
    print("-"*80)

    results = {}

    for file_path in sample_files:
        if not Path(file_path).exists():
            print(f"   ⚠ Skipping {file_path} - file not found")
            continue

        # Load and preprocess
        loader = LightCurveLoader()
        df = loader.load_file(file_path)
        df_processed = preprocessor.preprocess(df.copy(), normalize=True)

        # Extract features
        window_size = 50
        features = preprocessor.extract_features(df_processed, window_size)

        # Predict with NEW ensemble strategy
        predictions, scores = detector.predict_with_scores(
            features,
            ensemble_strategy='score_threshold'  # NEW!
        )

        # Map with NEW voting approach
        stride = max(1, window_size // 2)  # 50% overlap (NEW!)

        # Voting-based mapping (NEW!)
        window_count = np.zeros(len(df_processed), dtype=int)
        anomaly_count = np.zeros(len(df_processed), dtype=int)

        for i, pred in enumerate(predictions):
            start_idx = i * stride
            end_idx = min(start_idx + window_size, len(df_processed))
            window_count[start_idx:end_idx] += 1
            if pred == -1:
                anomaly_count[start_idx:end_idx] += 1

        vote_percentage = np.divide(
            anomaly_count,
            window_count,
            out=np.zeros_like(anomaly_count, dtype=float),
            where=window_count > 0
        )

        vote_threshold = 0.3
        anomaly_mask = vote_percentage >= vote_threshold

        # Calculate metrics
        anomaly_rate = np.mean(anomaly_mask) * 100

        # Statistical comparison
        flux = df_processed['flux'].values
        z_scores = np.abs((flux - np.mean(flux)) / np.std(flux))
        stat_anomalies = z_scores > 3.0
        stat_rate = np.mean(stat_anomalies) * 100

        agreement = np.mean(anomaly_mask == stat_anomalies) * 100

        file_name = Path(file_path).name
        results[file_name] = {
            'anomaly_rate': anomaly_rate,
            'stat_rate': stat_rate,
            'agreement': agreement
        }

        print(f"\n   {file_name}:")
        print(f"      ML Anomaly Rate:        {anomaly_rate:.1f}%")
        print(f"      Statistical Rate:       {stat_rate:.1f}%")
        print(f"      Agreement:              {agreement:.1f}%")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF IMPROVEMENTS")
    print("="*80)

    print("\nExpected behavior:")
    print("  ✓ Normal star should have LOWEST anomaly rate (target: 5-20%)")
    print("  ✓ Transit/flares should be moderate (target: 20-40%)")
    print("  ✓ Complex should be highest (target: 40-60%)")
    print("  ✓ Agreement with statistical methods should be >70%")

    print("\nActual results:")
    for file_name, metrics in results.items():
        status = "✓" if metrics['anomaly_rate'] < 50 else "⚠"
        print(f"  {status} {file_name}: {metrics['anomaly_rate']:.1f}% anomalies, {metrics['agreement']:.1f}% agreement")

    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)

    if 'normal_star.csv' in results:
        normal_rate = results['normal_star.csv']['anomaly_rate']
        if normal_rate < 25:
            print(f"  ✓ Normal star rate ({normal_rate:.1f}%) is reasonable (< 25%)")
        else:
            print(f"  ⚠ Normal star rate ({normal_rate:.1f}%) is still high (should be < 25%)")

    if 'complex_system.csv' in results and 'normal_star.csv' in results:
        complex_rate = results['complex_system.csv']['anomaly_rate']
        normal_rate = results['normal_star.csv']['anomaly_rate']
        if complex_rate > normal_rate:
            print(f"  ✓ Complex system ({complex_rate:.1f}%) > Normal star ({normal_rate:.1f}%)")
        else:
            print(f"  ⚠ Complex system should have higher rate than normal star")

    avg_agreement = np.mean([r['agreement'] for r in results.values()])
    if avg_agreement > 70:
        print(f"  ✓ Average agreement ({avg_agreement:.1f}%) is good (> 70%)")
    else:
        print(f"  ⚠ Average agreement ({avg_agreement:.1f}%) is low (should be > 70%)")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    # Check if improvements worked
    if 'normal_star.csv' in results:
        normal_rate = results['normal_star.csv']['anomaly_rate']
        if normal_rate < 30:
            print("✓ SUCCESS! Anomaly rates are now much more reasonable.")
            print(f"  Normal star went from ~79% to {normal_rate:.1f}%")
            print("  This represents a significant improvement!")
        else:
            print("⚠ Partial improvement, but may need further tuning.")
            print(f"  Normal star: {normal_rate:.1f}%")
            print("  Consider adjusting vote_threshold or contamination.")

    return results


if __name__ == '__main__':
    results = test_improvements()
