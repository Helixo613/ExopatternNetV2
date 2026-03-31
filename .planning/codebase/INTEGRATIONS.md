# INTEGRATIONS

## External Astronomy Data Sources
- NASA MAST archive is the primary light-curve source.
- `backend/data/acquisition.py` uses `lightkurve` search/download APIs to fetch Kepler and TESS light curves.
- `backend/data/acquisition.py` also depends on the NASA Exoplanet Archive via `astroquery.nasa_exoplanet_archive`.
- Ground-truth transit labels are derived from ephemerides in the Exoplanet Archive, not from local annotations.

## External Scientific Libraries
- `lightkurve` provides mission search, download, stitching, normalization, and light-curve handling.
- `astropy` is used for FITS IO and Box Least Squares support.
- `PyWavelets` provides continuous wavelet transform features in `backend/ml/preprocessing.py`.
- `scikit-learn` provides Isolation Forest, LOF, One-Class SVM, DBSCAN, scaling, CV, and metrics.
- `TensorFlow/Keras` provides the Conv1D and LSTM autoencoders in `backend/ml/deep_models.py`.

## Web And Service Endpoints
- Flask API endpoints are defined in `backend/api/routes.py`.
- `GET /health` reports service status and whether a model is loaded.
- `POST /api/analyze` accepts FITS or CSV files for scoring.
- `POST /api/train` trains on uploaded files.
- `POST /api/train/synthetic` initializes a baseline model from synthetic data.
- `POST /api/export` returns CSV results.
- `GET /api/models` lists available classical models.
- `POST /api/evaluate` scores labeled inputs.

## Frontend Integration Points
- Streamlit UI is implemented in `frontend/app.py`.
- The frontend uploads user files, loads model artifacts, runs preprocessing, and renders anomaly visualizations.
- The app depends on local pretrained artifacts from `artifacts/models/` and falls back to synthetic training if they are missing.
- Plotly charts are rendered client-side from Python-generated data.

## Local Artifact Integration
- Classical pretrained artifacts are read from `artifacts/models/*.pkl`.
- Deep model artifacts are read from `artifacts/models/*.h5` plus companion metadata JSON files.
- `backend/ml/model_registry.py` is the central loader for model artifacts.
- `backend/ml/training.py` writes model bundles into `backend/models/` for the default runtime path.

## Kaggle Workflow Integration
- The Kaggle notebooks under `notebooks/kaggle/` are the heavyweight training and data acquisition workflow.
- Notebook 01 downloads labeled data and writes it into a Kaggle dataset structure.
- Notebook 03 trains classical models and exports `.pkl` artifacts.
- Notebook 05 trains deep models and exports `.h5` artifacts.
- Notebook 04 generates the experiment figures and tables.

## File And Data Exchange
- CSV and FITS are the main interchange formats for light curves.
- `data/labeled/metadata.csv` ties together target metadata and per-star light-curve files.
- `data/labeled/lightcurves/*.csv` store star-level labeled sequences.
- `data/samples/*.csv` and `data/samples/*.fits` are synthetic examples for smoke testing.

## External Tooling
- `run.sh` bootstraps a local Python environment and starts the Streamlit app.
- `run_wsl.sh` applies WSL-specific Streamlit settings for browser access from Windows.
- `scripts/download_dataset.py` is the command-line data acquisition entry point.
- `generate_sample_data.py` creates local demo files without needing network access.

## Non-Existent Or Minimal Integrations
- There is no database integration in the current repo.
- There is no authentication provider or OAuth flow in the current codebase.
- There are no webhook integrations.
- There is no message queue or background job system.

## Integration Risks
- Internet access is required for live MAST and NASA Exoplanet Archive access.
- TensorFlow is only needed for deep models, but the repo imports it directly where the models are defined.
- File-path assumptions matter: local paths, Kaggle paths, and WSL paths are all handled differently.
- Model compatibility depends on package versions, especially scikit-learn, astropy, and TensorFlow.

