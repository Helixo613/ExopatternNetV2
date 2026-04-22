# ExopatternNet V3 — Stellar Light Curve Anomaly Detector

ML system for detecting exoplanet transits and stellar anomalies in astronomical time-series data. Benchmarks 9 models (classical + period-search baselines + deep learning) on real Kepler/TESS light curves with NASA ephemeris ground truth. Built for conference paper publication (v3.1 research pipeline).

## What It Does

- Downloads real stellar light curves from MAST (Kepler/TESS) via lightkurve
- Creates per-point ground truth labels from NASA Exoplanet Archive ephemerides
- Extracts 38 features per window across 5 feature groups (statistical, frequency, wavelet, autocorrelation, shape)
- Trains and evaluates 9 models with 5-fold cross-validation including domain-specific period-search baselines (BLS, TLS)
- Period-aware propagation: BLS/TLS detections propagate across orbital periods for fair per-point evaluation
- V3 anomaly ranking pipeline with injection detection and cross-fold ablation
- Generates publication-quality figures and LaTeX tables
- Provides a Streamlit frontend for interactive analysis

## Project Structure

```
backend/
  data/
    loader.py              # Universal FITS/CSV loader
    acquisition.py         # MAST data download + ground truth labeling
  ml/
    models.py              # Core AnomalyDetector (IF+LOF ensemble)
    preprocessing.py       # Feature extraction (38 features, 5 groups)
    feature_names.py       # Feature name registry
    baselines.py           # 6 classical models (IF, LOF, OCSVM, DBSCAN, Ensemble, BLS)
    deep_models.py         # CNN + LSTM autoencoders (TensorFlow)
    model_registry.py      # Model factory registry
    evaluation.py          # Metrics, CV, bootstrap CIs, paired tests
    figures.py             # Publication figure generator (matplotlib)
    training.py            # Model trainer with synthetic data support
  api/routes.py            # Flask REST API
  app.py                   # Flask entry point
frontend/app.py            # Streamlit UI
experiments/
  config.py                # Experiment constants + hyperparameter grids
  run_experiments.py       # 7 experiments for paper results
  hyperparameter_tuning.py # Grid search
scripts/
  download_dataset.py      # CLI for MAST data download
notebooks/kaggle/          # 5 Kaggle notebooks for full-scale runs
data/
  samples/                 # Synthetic test data (CSV + FITS)
  labeled/                 # Downloaded Kepler data with labels
results/
  figures/                 # PDF figures
  tables/                  # LaTeX tables
  raw/                     # Raw JSON metrics
```

## Quick Start

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run frontend/app.py
# WSL: ./run_wsl.sh
```

### Download Real Data (Smoke Test)

```bash
python scripts/download_dataset.py --smoke-test --verbose
```

Downloads 5 Kepler planet hosts (3 quarters each) to `data/labeled/`.

### Run Experiments

```bash
python experiments/run_experiments.py --experiment 1  # Model comparison
python experiments/run_experiments.py --experiment 0  # All 7 experiments
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| Isolation Forest | Classical | Tree-based anomaly isolation |
| LOF | Classical | Local density-based outlier detection |
| One-Class SVM | Classical | Kernel-based novelty detection |
| DBSCAN | Classical | Density clustering (outliers = unclustered) |
| Ensemble (IF+LOF) | Classical | AND-logic ensemble requiring both models to agree |
| BLS Detector | Domain | Box Least Squares transit search with period-aware propagation |
| TLS Detector | Domain | Transit Least Squares — fast direct per-star evaluation |
| Conv1D Autoencoder | Deep Learning | 1D-CNN reconstruction error |
| LSTM Autoencoder | Deep Learning | Sequence-based reconstruction error |

## Feature Groups (38 total)

| Group | Count | Features |
|-------|-------|----------|
| Statistical | 14 | mean, std, median, min, max, ptp, skew, kurtosis, slope, intercept, MAD, RMS, mean/max abs diff |
| Frequency | 7 | Top 3 FFT amplitudes, dominant frequency, spectral entropy/centroid/rolloff |
| Wavelet | 7 | CWT energy at 5 scales, wavelet entropy, max-energy scale |
| Autocorrelation | 5 | ACF at lags 1/5/10, first zero crossing, decay rate |
| Shape | 5 | Asymmetry, flatness, edge gradient ratio, min position, beyond-1sigma fraction |

## Experiments

1. **Model Comparison** — All models x 5-fold CV with precision/recall/F1/ROC-AUC
2. **Feature Ablation** — Cumulative feature groups on best model (parallelized CV folds)
3. **Per-Type Detection** — Metrics by anomaly type
4. **Hyperparameter Sensitivity** — Contamination and window size sweeps
5. **Detection Examples** — Qualitative light curve figures with injection feature caching
6. **BLS Baseline** — Period-aware BLS propagation; fair per-point transit detection comparison
7. **TLS Baseline** — Transit Least Squares with fast direct per-star evaluation

## Kaggle Workflow

Heavy compute runs on Kaggle. Local is code + Streamlit with pre-trained artifacts.

1. Upload `backend/` + `experiments/` as Kaggle dataset `exopattern-codebase`
2. Run `01_data_acquisition.ipynb` (internet ON) — downloads 150 targets
3. Run `03_model_training.ipynb` — trains classical models, exports `.pkl`
4. Run `05_deep_learning.ipynb` (GPU ON) — trains autoencoders, exports `.h5`
5. Run `04_experiments.ipynb` — generates all paper figures and tables
6. Download artifacts to local `artifacts/` directory

## Key Dependencies

numpy, pandas, scikit-learn, scipy, astropy, lightkurve, astroquery, PyWavelets, matplotlib, tensorflow, streamlit, plotly, flask

## References

- WaldoInSky (github.com/kushaltirumala/WaldoInSky)
- StellarScope (github.com/Fastian-afk/Stellar-Scope)
- NASA Kepler/TESS missions, NASA Exoplanet Archive
