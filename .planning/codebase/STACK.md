# STACK

## Runtime
- Primary language: Python.
- Local execution targets Python 3.12 in the checked-in virtual environment at `venv/pyvenv.cfg`.
- The app is launched either through `run.sh`, `run_wsl.sh`, `backend/app.py`, or `streamlit run frontend/app.py`.
- The repo is organized as a Python package-style monorepo with `backend/`, `frontend/`, `experiments/`, and `scripts/`.

## Application Frameworks
- UI: Streamlit in `frontend/app.py`.
- API: Flask plus Flask-CORS in `backend/api/routes.py` and `backend/app.py`.
- Plotting/UI charts: Plotly in the Streamlit app and matplotlib in `backend/ml/figures.py`.
- Notebook workflow: Kaggle Jupyter notebooks under `notebooks/kaggle/`.

## Core ML Stack
- Data frames and numeric work: NumPy and pandas.
- Classical anomaly detection: scikit-learn in `backend/ml/baselines.py` and `backend/ml/models.py`.
- Signal processing and stats: SciPy in `backend/ml/preprocessing.py` and `backend/ml/evaluation.py`.
- Astronomy-specific IO and archives: Astropy, lightkurve, and astroquery in `backend/data/loader.py` and `backend/data/acquisition.py`.
- Deep learning: TensorFlow/Keras in `backend/ml/deep_models.py`.
- Persistence: joblib for classical model artifacts in `backend/ml/models.py` and `backend/ml/model_registry.py`.

## Domain Feature Stack
- Wavelet features: PyWavelets in `backend/ml/preprocessing.py`.
- Transit-search features and periodograms: Lomb-Scargle, Box Least Squares, and related astronomy utilities in `backend/ml/preprocessing.py`.
- Publication artifacts: PDF/LaTeX generation in `backend/ml/figures.py`.

## Command-Line Entry Points
- `scripts/download_dataset.py` downloads and labels Kepler/TESS data.
- `generate_sample_data.py` creates synthetic light curves for smoke tests.
- `experiments/run_experiments.py` runs the paper experiment suite.
- `experiments/hyperparameter_tuning.py` performs grid search.

## Configuration Files
- Dependency pins live in `requirements.txt`.
- Streamlit server and theme settings live in `.streamlit/config.toml`.
- Shell launch behavior and dependency bootstrap live in `run.sh` and `run_wsl.sh`.
- Experiment constants live in `experiments/config.py`.

## Model Artifacts
- Classical model pickles are stored under `backend/models/`.
- Larger pretrained artifacts live in `artifacts/models/` and are loaded by `backend/ml/model_registry.py`.
- Example outputs and packaged notebooks live in `notebooks/outputs/`.

## Data Layout
- Synthetic smoke-test data lives in `data/samples/`.
- Labeled Kepler data lives in `data/labeled/`.
- Generated analysis outputs live in `results/`.
- The Flask upload scratch directory is `backend/uploads/`.

## Build And Run Characteristics
- There is no compiled build step; everything is interpreted Python.
- The main runtime assumption is a local environment with scientific Python packages installed.
- Kaggle notebooks are used for heavy training and data acquisition, while the local repo serves the UI and inference paths.
- GPU is optional and only relevant for TensorFlow-based models.

## Dependency Profile
- This is a research ML stack, not a web-first application stack.
- Astronomy packages are first-class dependencies, not optional add-ons.
- The repo depends on a mix of scientific computing, web serving, and interactive visualization libraries.
- Most packages are imported directly in application code rather than hidden behind plugin abstractions.

