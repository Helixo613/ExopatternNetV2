# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stellar Light Curve Anomaly Detector - ML-based system for detecting exoplanet transits, stellar flares, and anomalous patterns in astronomical time-series data. Uses ensemble anomaly detection (Isolation Forest + LOF + statistical methods) with a Streamlit frontend.

## Common Commands

### Running the Application

**Primary method (Streamlit standalone):**
```bash
streamlit run frontend/app.py
```

**WSL/Linux quick launcher:**
```bash
./run.sh          # Standard Linux/Mac
./run_wsl.sh      # WSL-optimized (binds to 0.0.0.0 for Windows access)
```

**Backend API (optional - for REST API access):**
```bash
python backend/app.py  # Starts Flask API on port 5000
```

### Data Generation

```bash
# Generate sample light curves (5 types: normal, transit, flare, outliers, complex)
python generate_sample_data.py --format both --n-samples 5

# Custom generation
python generate_sample_data.py --output-dir data/samples --format csv --n-samples 10
```

### Testing with Sample Data

Sample files located in `data/samples/`:
- `exoplanet_transit.csv` - Best for testing transit detection
- `stellar_flares.csv` - Tests flare/spike detection
- `complex_system.csv` - Tests all anomaly types
- `normal_star.csv` - Control (should have <5% anomalies)
- All available in both CSV and FITS formats

## Architecture & Data Flow

### Three-Layer Architecture

**1. Data Layer (`backend/data/`)**
- `LightCurveLoader` - Universal loader for FITS (Kepler/TESS) and CSV formats
- Auto-detects column names (time/TIME, flux/FLUX/SAP_FLUX, flux_err/FLUX_ERR)
- Validates data, removes NaN/inf, sorts by time
- Returns pandas DataFrame with standardized columns: `time`, `flux`, `flux_err`

**2. ML Layer (`backend/ml/`)**
- **`LightCurvePreprocessor`**:
  - Sigma clipping (5σ outlier removal)
  - Normalization (zero mean, unit variance)
  - Gap filling via interpolation
  - Feature extraction: Sliding window → 14 features/window (statistics, trends, variability)

- **`AnomalyDetector`**:
  - Ensemble of Isolation Forest + LOF (both trained, predictions combined)
  - Three detection modes: window-based, point-based (z-score), event-based (transits/flares)
  - Models saved via joblib to `backend/models/`

- **`ModelTrainer`**:
  - Trains from multiple files or single DataFrame
  - Can generate synthetic training data (creates realistic light curves with injected anomalies)
  - Handles model persistence

**3. API/Frontend Layer**
- **Flask API (`backend/api/routes.py`)**: REST endpoints for programmatic access
- **Streamlit (`frontend/app.py`)**: Primary user interface with visualization

### Critical Data Flow Pattern

```
Upload → LightCurveLoader.load_file()
       → LightCurvePreprocessor.preprocess()
       → LightCurvePreprocessor.extract_features(window_size=50)
       → AnomalyDetector.predict_with_scores()
       → Map window predictions back to original points (stride = window_size // 4)
       → Visualization with anomaly highlighting
```

**Key detail:** Window-based predictions are mapped to original data points using overlapping windows (75% overlap). Each point can be marked anomalous if it falls in ANY anomalous window.

## Important Implementation Details

### Feature Extraction Windows

The sliding window approach is central to detection:
- Default `window_size=50` points
- Stride = `window_size // 4` (75% overlap)
- Each window → 14 features: mean, std, median, min, max, ptp, skew, kurtosis, slope, intercept, MAD, RMS, mean_abs_diff, max_abs_diff

Adjust window_size based on anomaly type:
- Transits (1-3 hours): 50-100 points
- Flares (minutes): 10-30 points
- Long trends: 100-200 points

### Model Training State

Models are stateful. The `AnomalyDetector` must be fitted before prediction:

```python
detector = AnomalyDetector(contamination=0.1)
detector.fit(features)  # REQUIRED before predict()
detector.is_fitted  # Check this flag
```

In Streamlit app, model is stored in `st.session_state.detector` and initialized on first use or via "Initialize Model" button.

### File Format Handling

**CSV:** Flexible column detection tries: `time`/`TIME`/`JD`/`MJD`, `flux`/`FLUX`/`SAP_FLUX`
- If columns not found, auto-maps first 3 columns → time, flux, flux_err

**FITS:** Looks for `LIGHTCURVE` extension or first binary table
- Extracts metadata from primary header (OBJECT, TELESCOP, INSTRUME)
- Supports Kepler/TESS standard formats

### WSL-Specific Considerations

When running in WSL (Windows Subsystem for Linux):
- Streamlit must bind to `0.0.0.0` not `127.0.0.1` for Windows browser access
- Use `run_wsl.sh` which sets `--server.address=0.0.0.0`
- Or create `.streamlit/config.toml` with `address = "0.0.0.0"`
- VS Code port forwarding may auto-forward port 8501

## Extension Points

**Adding new detection algorithms:**
1. Add detector class to `backend/ml/models.py`
2. Implement `fit()` and `predict()` methods matching sklearn API
3. Add to `EnsembleAnomalyDetector` or use standalone

**Adding new features:**
1. Extend `LightCurvePreprocessor.extract_features()` in `backend/ml/preprocessing.py`
2. Add feature computation in sliding window loop
3. Features automatically flow to all detectors

**Adding new file formats:**
1. Add loader method to `LightCurveLoader` in `backend/data/loader.py`
2. Return standardized DataFrame (time, flux, flux_err columns)
3. Add file extension check in `load_file()`

**Adding new visualizations:**
1. Add plot function to `frontend/app.py`
2. Use Plotly for consistency (already imported as `go`)
3. Call from appropriate tab (Analyze/Statistics/etc.)

## Critical Gotchas

1. **Import Path:** Both `frontend/app.py` and `backend/app.py` add parent directory to `sys.path`. If imports fail, verify working directory is project root.

2. **Nonlocal Variable in Flask:** `backend/api/routes.py` uses `nonlocal detector` in nested functions to allow model updates. This is intentional for stateful model management.

3. **Model Directory Must Exist:** `backend/models/` and `backend/uploads/` must exist (tracked via `.gitkeep`). Scripts create them, but if deleted, mkdir first.

4. **Contamination Parameter:** This is NOT a threshold - it's the expected proportion of anomalies (0.1 = 10%). Higher values make detector MORE sensitive (flags more as anomalous).

5. **Window Predictions ≠ Point Predictions:** The detector predicts on feature windows, not individual points. Must map back to points using the stride pattern. See `analyze_lightcurve()` in frontend for mapping logic.

6. **Synthetic Data Seeds:** `generate_sample_data.py` uses fixed random seeds per sample type for reproducibility. Same file name always generates identical data.

## Performance Characteristics

- **File loading**: <1s for typical 2000-point light curve
- **Preprocessing + feature extraction**: 2-5s
- **Model training (synthetic)**: 10-30s for 100 curves
- **Prediction**: <1s
- **Memory**: ~200MB baseline, ~1-5MB per light curve, ~10-50MB per model

Tested up to 100k points and 1000 training light curves. Not designed for streaming or high-concurrency deployment.

## Configuration Files

**`.streamlit/config.toml`** (created by debugger agent for WSL):
```toml
[server]
address = "0.0.0.0"  # Required for WSL
enableCORS = false
enableXsrfProtection = false
```

**`requirements.txt`**: All dependencies pinned to tested versions. Core: numpy, pandas, scikit-learn, astropy (FITS), streamlit, plotly, flask.

## Sample Data Details

Five generated samples (see `generate_sample_data.py`):
1. **normal_star**: Pure stellar variability (3 sine waves + noise)
2. **exoplanet_transit**: 15-day period, 30-unit depth, box-like dips
3. **stellar_flares**: 2-4 random exponential spikes
4. **noisy_outliers**: 10-30 random outliers ±50-150 units
5. **complex_system**: All of the above combined

Each is 2000 points over 100 days, flux ~1000 ± 5-15 units std.

## References

Inspired by:
- WaldoInSky (github.com/kushaltirumala/WaldoInSky) - Multi-algorithm anomaly comparison
- StellarScope (github.com/Fastian-afk/Stellar-Scope) - Gradio-based analyzer

Detection methods based on established ML approaches (Isolation Forest, LOF) adapted for time-series astronomy.
