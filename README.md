# Stellar Light Curve Anomaly Detector

An ML-based system for detecting exoplanet transits, stellar flares, and anomalous patterns in astronomical time-series data. Uses ensemble anomaly detection (Isolation Forest + LOF) with a Streamlit frontend.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What It Does

- Detects **exoplanet transits** (periodic brightness dips)
- Identifies **stellar flares** (sudden brightness spikes)
- Flags **instrumental artifacts** and outliers
- Supports **FITS** (Kepler/TESS) and **CSV** formats

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Helixo613/ExopatternNetV2.git
cd ExopatternNetV2

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

**WSL Users:** Use `./run_wsl.sh` or add `--server.address=0.0.0.0` to access from Windows browser.

## Usage

### Analyzing a Light Curve

1. Open the app (`streamlit run frontend/app.py`)
2. Upload a FITS or CSV file in the Analyze tab
3. Click "Analyze Light Curve"
4. View results - anomalies are highlighted in red on the plot

### Training a Custom Model

Go to the **Train Model** tab:

- **Quick start:** Select "Synthetic Data", set sample count to 100, click Train
- **Custom data:** Upload your own light curve files and train on those

### Configuration Options

Adjust in the sidebar:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Contamination | 5% | Expected anomaly rate. Higher = more sensitive |
| Window Size | 50 | Analysis window in data points. Larger = catches longer events |
| Vote Threshold | 0.3 | Fraction of windows that must agree to flag a point |

## File Formats

### CSV

```csv
time,flux,flux_err
0.0,1000.5,2.3
0.1,1002.1,2.1
```

Columns auto-detected. Accepts: `time`/`TIME`/`JD`/`MJD`, `flux`/`FLUX`/`SAP_FLUX`

### FITS

Standard Kepler/TESS format. Looks for `LIGHTCURVE` extension or first binary table.

## Sample Data

Generate test light curves:

```bash
python generate_sample_data.py --format both --n-samples 5
```

Creates 5 sample files in `data/samples/`:

| File | Contains |
|------|----------|
| `normal_star.csv` | Clean baseline (stellar variability only) |
| `exoplanet_transit.csv` | Periodic transit dips every 15 days |
| `stellar_flares.csv` | Random flare events |
| `noisy_outliers.csv` | Instrumental artifacts |
| `complex_system.csv` | All of the above combined |

## Project Structure

```
ExopatternNetV2/
├── backend/
│   ├── api/          # Flask REST API
│   ├── data/         # Data loaders (FITS, CSV)
│   ├── ml/           # ML models and preprocessing
│   └── models/       # Saved model files
├── frontend/
│   └── app.py        # Streamlit UI
├── data/samples/     # Sample light curves
└── requirements.txt
```

## How It Works

1. **Load** - `LightCurveLoader` reads FITS/CSV, standardizes columns
2. **Preprocess** - Sigma clipping, normalization, gap filling
3. **Extract Features** - Sliding window (50 points) → 14 statistical features per window
4. **Detect** - Ensemble of Isolation Forest + LOF with score-based thresholding
5. **Map** - Window predictions mapped back to points via voting

The ensemble uses a score-threshold strategy (not simple OR logic) to reduce false positives. Points are only flagged if multiple overlapping windows agree.

## Python API

Use the backend directly:

```python
from backend.data import LightCurveLoader
from backend.ml import LightCurvePreprocessor, AnomalyDetector

# Load and preprocess
loader = LightCurveLoader()
df = loader.load_file('data/samples/exoplanet_transit.csv')

preprocessor = LightCurvePreprocessor()
df_processed = preprocessor.preprocess(df)
features = preprocessor.extract_features(df_processed, window_size=50)

# Train and predict
detector = AnomalyDetector(contamination=0.05)
detector.fit(features)
predictions, scores = detector.predict_with_scores(features)

# Save for later
detector.save_model('backend/models/my_model')
```

## REST API

Start the backend separately:

```bash
python backend/app.py  # Runs on port 5000
```

Endpoints:

```bash
# Analyze a file
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@lightcurve.csv" \
  -F "contamination=0.05"

# Train on synthetic data
curl -X POST http://localhost:5000/api/train/synthetic \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 100, "contamination": 0.05}'

# Health check
curl http://localhost:5000/health
```

## Troubleshooting

**Import errors:** Run from project root directory

**Model not loading:** Train a new model from the Train tab

**FITS errors:** Install astropy: `pip install astropy`

**WSL can't access app:** Use `--server.address=0.0.0.0` or run `./run_wsl.sh`

## References

Inspired by:
- [WaldoInSky](https://github.com/kushaltirumala/WaldoInSky) - Anomaly detection algorithm comparison
- [StellarScope](https://github.com/Fastian-afk/Stellar-Scope) - Kepler light curve analysis

## License

MIT
