# 📚 Usage Examples

This guide provides detailed examples of using the Stellar Light Curve Anomaly Detector.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Python API Examples](#python-api-examples)
3. [REST API Examples](#rest-api-examples)
4. [Advanced Workflows](#advanced-workflows)

---

## Basic Usage

### Example 1: Analyze a Light Curve with Exoplanet Transit

```bash
# Generate sample data
python generate_sample_data.py

# Launch the app
streamlit run frontend/app.py
```

1. Click "Initialize/Reload Model"
2. Upload `data/samples/exoplanet_transit.csv`
3. Click "Analyze Light Curve"
4. **Expected Result**: 5-6 transit events detected, ~15-day period

### Example 2: Detect Stellar Flares

1. Upload `data/samples/stellar_flares.csv`
2. Adjust "Expected Anomaly Rate" to 0.15 (15%)
3. Analyze
4. **Expected Result**: Multiple spike anomalies, check "Point Anomalies" → "Flux Spikes"

### Example 3: Train on Custom Data

1. Go to "Train Model" tab
2. Upload 5-10 of your own light curve files
3. Set contamination to 0.1
4. Click "Train Model"
5. Go back to "Analyze" and test on new data

---

## Python API Examples

### Example 1: Load and Visualize Data

```python
from backend.data import LightCurveLoader
import matplotlib.pyplot as plt

# Load a light curve
loader = LightCurveLoader()
df = loader.load_file('data/samples/exoplanet_transit.csv')

# Get statistics
stats = loader.get_summary_stats(df)
print(f"Loaded {stats['n_points']} points")
print(f"Time span: {stats['time_span']:.2f} days")

# Plot
plt.figure(figsize=(12, 4))
plt.plot(df['time'], df['flux'], 'b.', markersize=2)
plt.xlabel('Time (days)')
plt.ylabel('Flux')
plt.title('Light Curve')
plt.show()
```

### Example 2: Preprocess and Extract Features

```python
from backend.ml import LightCurvePreprocessor
import pandas as pd

# Load data
df = pd.read_csv('data/samples/stellar_flares.csv')

# Initialize preprocessor
preprocessor = LightCurvePreprocessor()

# Preprocess
df_clean = preprocessor.preprocess(df, normalize=True)

# Extract features
features = preprocessor.extract_features(df_clean, window_size=50)
print(f"Extracted features shape: {features.shape}")
print(f"Features per window: {features.shape[1]}")

# Smooth the light curve
smoothed_flux = preprocessor.smooth_lightcurve(df_clean['flux'].values)
```

### Example 3: Train and Use Anomaly Detector

```python
from backend.ml import AnomalyDetector, LightCurvePreprocessor
import pandas as pd
import numpy as np

# Load and prepare data
df = pd.read_csv('data/samples/complex_system.csv')
preprocessor = LightCurvePreprocessor()
df_clean = preprocessor.preprocess(df, normalize=True)
features = preprocessor.extract_features(df_clean, window_size=50)

# Create and train detector
detector = AnomalyDetector(contamination=0.1)
detector.fit(features)

# Predict anomalies
predictions, scores = detector.predict_with_scores(features)

# Analyze results
n_anomalies = np.sum(predictions == -1)
print(f"Detected {n_anomalies} anomalous windows")
print(f"Anomaly rate: {n_anomalies / len(predictions) * 100:.1f}%")

# Find most anomalous windows
anomaly_indices = np.where(predictions == -1)[0]
most_anomalous = anomaly_indices[np.argsort(scores[anomaly_indices])[:5]]
print(f"Most anomalous windows: {most_anomalous}")

# Save model
detector.save_model('backend/models/my_detector')
```

### Example 4: Detect Specific Event Types

```python
from backend.ml import AnomalyDetector
import pandas as pd

df = pd.read_csv('data/samples/exoplanet_transit.csv')

# Create detector
detector = AnomalyDetector()

# Detect point anomalies
point_anomalies = detector.detect_point_anomalies(df, threshold=3.0)
print(f"Point anomalies detected: {point_anomalies['n_anomalies']}")
print(f"Dips: {point_anomalies['n_dips']}, Spikes: {point_anomalies['n_spikes']}")

# Detect transit events
transit_events = detector.detect_transit_events(
    df,
    depth_threshold=0.01,  # 1% depth
    duration_min=3         # At least 3 consecutive points
)

print(f"\nTransit events detected: {len(transit_events)}")
for i, event in enumerate(transit_events):
    print(f"Event {i+1}:")
    print(f"  Time: {event['start_time']:.2f} - {event['end_time']:.2f}")
    print(f"  Depth: {event['depth']*100:.2f}%")
    print(f"  Duration: {event['time_duration']:.2f} days")
```

### Example 5: Complete Analysis Pipeline

```python
from backend.data import LightCurveLoader
from backend.ml import LightCurvePreprocessor, AnomalyDetector, ModelTrainer
import pandas as pd

# Step 1: Load data
loader = LightCurveLoader()
df = loader.load_file('path/to/your/lightcurve.fits')
print(f"Loaded {len(df)} data points")

# Step 2: Train or load model
trainer = ModelTrainer()
try:
    detector = trainer.load_model('default_model')
    print("Loaded existing model")
except:
    print("Training new model...")
    detector = trainer.train_from_dataframe(df, contamination=0.1, window_size=50)

# Step 3: Preprocess
preprocessor = LightCurvePreprocessor()
df_clean = preprocessor.preprocess(df, normalize=True)

# Step 4: Extract features
features = preprocessor.extract_features(df_clean, window_size=50)

# Step 5: Detect anomalies
predictions, scores = detector.predict_with_scores(features)

# Step 6: Map to original data points
stride = 50 // 4
anomaly_mask = np.zeros(len(df_clean), dtype=bool)
for i, pred in enumerate(predictions):
    if pred == -1:
        start_idx = i * stride
        end_idx = min(start_idx + 50, len(df_clean))
        anomaly_mask[start_idx:end_idx] = True

# Step 7: Create results dataframe
results = df_clean.copy()
results['is_anomaly'] = anomaly_mask

# Step 8: Export
results.to_csv('analysis_results.csv', index=False)
print(f"Analysis complete! Detected {anomaly_mask.sum()} anomalous points")
```

---

## REST API Examples

### Example 1: Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Example 2: Analyze Light Curve

```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@data/samples/exoplanet_transit.csv" \
  -F "contamination=0.1" \
  -F "method=ensemble" \
  -F "window_size=50"
```

**Response:**
```json
{
  "success": true,
  "filename": "exoplanet_transit.csv",
  "n_points": 2000,
  "statistics": {
    "n_points": 2000,
    "time_span": 100.0,
    "flux_mean": 1000.5,
    ...
  },
  "anomalies": {
    "window_based": {
      "n_anomalous_windows": 45,
      "anomaly_rate": 0.12,
      ...
    },
    "transit_events": [...]
  }
}
```

### Example 3: Train Model on Synthetic Data

```bash
curl -X POST http://localhost:5000/api/train/synthetic \
  -H "Content-Type: application/json" \
  -d '{
    "n_samples": 100,
    "contamination": 0.1
  }'
```

### Example 4: Train Model on Real Data

```bash
curl -X POST http://localhost:5000/api/train \
  -F "files=@data/samples/normal_star.csv" \
  -F "files=@data/samples/exoplanet_transit.csv" \
  -F "contamination=0.1" \
  -F "window_size=50"
```

---

## Advanced Workflows

### Workflow 1: Batch Processing Multiple Light Curves

```python
from pathlib import Path
import pandas as pd
from backend.data import LightCurveLoader
from backend.ml import LightCurvePreprocessor, AnomalyDetector, ModelTrainer

# Load model
trainer = ModelTrainer()
detector = trainer.load_model('default_model')
preprocessor = LightCurvePreprocessor()

# Process all CSV files in a directory
data_dir = Path('data/samples')
results_list = []

for file_path in data_dir.glob('*.csv'):
    print(f"Processing {file_path.name}...")

    try:
        # Load and preprocess
        loader = LightCurveLoader()
        df = loader.load_file(str(file_path))
        df_clean = preprocessor.preprocess(df, normalize=True)

        # Extract features and detect
        features = preprocessor.extract_features(df_clean, window_size=50)
        predictions, scores = detector.predict_with_scores(features)

        # Store results
        n_anomalies = (predictions == -1).sum()
        results_list.append({
            'filename': file_path.name,
            'n_points': len(df),
            'n_anomalies': n_anomalies,
            'anomaly_rate': n_anomalies / len(predictions),
            'mean_anomaly_score': scores.mean()
        })

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

# Create summary dataframe
summary = pd.DataFrame(results_list)
summary.to_csv('batch_processing_summary.csv', index=False)
print("\nBatch processing complete!")
print(summary)
```

### Workflow 2: Comparing Multiple Detection Methods

```python
from backend.ml import AnomalyDetector
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/samples/complex_system.csv')
from backend.ml import LightCurvePreprocessor
preprocessor = LightCurvePreprocessor()
df_clean = preprocessor.preprocess(df, normalize=True)
features = preprocessor.extract_features(df_clean, window_size=50)

# Create detector
detector = AnomalyDetector(contamination=0.1)
detector.fit(features)

# Compare methods
methods = ['isolation_forest', 'lof', 'ensemble']
results = {}

for method in methods:
    predictions = detector.predict(features, method=method)
    n_anomalies = (predictions == -1).sum()
    results[method] = {
        'n_anomalies': n_anomalies,
        'anomaly_rate': n_anomalies / len(predictions)
    }

# Print comparison
print("Method Comparison:")
for method, stats in results.items():
    print(f"{method:20s}: {stats['n_anomalies']:4d} anomalies ({stats['anomaly_rate']*100:.1f}%)")
```

### Workflow 3: Custom Feature Engineering

```python
from backend.ml import LightCurvePreprocessor
import pandas as pd
import numpy as np
from scipy import stats

class CustomPreprocessor(LightCurvePreprocessor):
    """Extended preprocessor with custom features."""

    def extract_custom_features(self, df, window_size=50):
        """Extract additional custom features."""
        flux = df['flux'].values
        time = df['time'].values

        features_list = []
        stride = max(1, window_size // 4)

        for i in range(0, len(flux) - window_size + 1, stride):
            window_flux = flux[i:i + window_size]
            window_time = time[i:i + window_size]

            # Standard features
            basic = [
                np.mean(window_flux),
                np.std(window_flux),
                stats.skew(window_flux),
                stats.kurtosis(window_flux)
            ]

            # Custom features
            # 1. Autocorrelation at lag 1
            if len(window_flux) > 1:
                autocorr = np.corrcoef(window_flux[:-1], window_flux[1:])[0, 1]
            else:
                autocorr = 0

            # 2. Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(window_flux - np.mean(window_flux))) != 0)

            # 3. Peak count
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(window_flux)
            n_peaks = len(peaks)

            custom = [autocorr, zero_crossings, n_peaks]

            features_list.append(basic + custom)

        return np.array(features_list)

# Use custom preprocessor
preprocessor = CustomPreprocessor()
df = pd.read_csv('data/samples/stellar_flares.csv')
df_clean = preprocessor.preprocess(df, normalize=True)
features = preprocessor.extract_custom_features(df_clean, window_size=50)

print(f"Extracted custom features: {features.shape}")
```

### Workflow 4: Time-Series Cross-Validation

```python
from backend.ml import AnomalyDetector
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/samples/complex_system.csv')
from backend.ml import LightCurvePreprocessor
preprocessor = LightCurvePreprocessor()
df_clean = preprocessor.preprocess(df, normalize=True)
features = preprocessor.extract_features(df_clean, window_size=50)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
contamination_values = [0.05, 0.1, 0.15, 0.2]
results = []

for contamination in contamination_values:
    fold_results = []

    for train_idx, test_idx in tscv.split(features):
        # Split data
        X_train = features[train_idx]
        X_test = features[test_idx]

        # Train
        detector = AnomalyDetector(contamination=contamination)
        detector.fit(X_train)

        # Predict
        predictions = detector.predict(X_test, method='ensemble')
        anomaly_rate = (predictions == -1).sum() / len(predictions)

        fold_results.append(anomaly_rate)

    mean_rate = np.mean(fold_results)
    std_rate = np.std(fold_results)

    results.append({
        'contamination': contamination,
        'mean_anomaly_rate': mean_rate,
        'std_anomaly_rate': std_rate
    })

# Print results
results_df = pd.DataFrame(results)
print("\nCross-validation Results:")
print(results_df)
```

---

## Tips and Best Practices

### 1. Choosing Window Size

- **Short transits (< 0.5 days)**: Use `window_size=20-30`
- **Typical exoplanet transits (1-3 hours)**: Use `window_size=50-100`
- **Stellar flares**: Use `window_size=10-30`
- **Long-term trends**: Use `window_size=100-200`

### 2. Adjusting Contamination

Start with 0.1 (10%) and adjust based on results:
- Too many false positives → Decrease contamination
- Missing known anomalies → Increase contamination

### 3. Data Quality

- Remove known bad data sections before analysis
- Check for gaps in time series
- Ensure consistent time units
- Verify flux units and normalization

### 4. Model Training

- Use at least 10-20 light curves for training
- Include diverse examples (normal + anomalous)
- Retrain periodically with new validated data
- Save multiple model versions for comparison

### 5. Validation

- Always visually inspect detected anomalies
- Compare with known events (if available)
- Use multiple detection methods
- Check false positive rate on control data

---

## Common Use Cases

### Exoplanet Transit Search
```python
detector = AnomalyDetector(contamination=0.05)  # Low contamination
transit_events = detector.detect_transit_events(
    df,
    depth_threshold=0.005,  # 0.5% depth
    duration_min=5
)
```

### Stellar Flare Detection
```python
detector = AnomalyDetector(contamination=0.15)  # Higher contamination
point_anomalies = detector.detect_point_anomalies(df, threshold=4.0)
flares = point_anomalies['spike_indices']
```

### Data Quality Assessment
```python
# Use high contamination to find all potential issues
detector = AnomalyDetector(contamination=0.3)
# Review all flagged points for instrumental artifacts
```

---

For more information, see [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).
