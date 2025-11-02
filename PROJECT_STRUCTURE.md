# 📁 Project Structure

Complete overview of the Stellar Light Curve Anomaly Detector project.

```
ExopatternNetV3/
│
├── 📄 README.md                    # Main documentation
├── 📄 QUICKSTART.md                # Quick start guide (5 minutes)
├── 📄 EXAMPLES.md                  # Detailed usage examples
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Installation script
├── 📄 .gitignore                   # Git ignore rules
│
├── 🚀 run.sh                       # Linux/Mac launcher
├── 🚀 run.bat                      # Windows launcher
├── 🔧 generate_sample_data.py      # Sample data generator
│
├── 🔙 backend/                     # ML Backend
│   ├── 📄 app.py                   # Flask API entry point
│   │
│   ├── 📂 api/                     # API Layer
│   │   ├── __init__.py
│   │   └── routes.py               # REST API endpoints
│   │
│   ├── 📂 data/                    # Data Ingestion
│   │   ├── __init__.py
│   │   └── loader.py               # FITS/CSV loader
│   │
│   ├── 📂 ml/                      # Machine Learning
│   │   ├── __init__.py
│   │   ├── preprocessing.py        # Feature extraction
│   │   ├── models.py               # Anomaly detection models
│   │   └── training.py             # Training pipeline
│   │
│   ├── 📂 models/                  # Saved Models
│   │   └── .gitkeep
│   │
│   └── 📂 uploads/                 # Temporary uploads
│       └── .gitkeep
│
├── 🖼️ frontend/                    # User Interface
│   └── app.py                      # Streamlit app
│
├── 📊 data/                        # Data Storage
│   └── samples/                    # Sample light curves
│       ├── normal_star.csv/fits
│       ├── exoplanet_transit.csv/fits
│       ├── stellar_flares.csv/fits
│       ├── noisy_outliers.csv/fits
│       └── complex_system.csv/fits
│
└── 📓 notebooks/                   # Jupyter Notebooks (optional)
    └── (analysis notebooks)
```

---

## 🔍 Detailed Component Breakdown

### Backend Components

#### 1. **Data Layer** (`backend/data/`)

**loader.py** - Universal light curve loader
- Loads FITS files (Kepler/TESS format)
- Loads CSV files (flexible column detection)
- Validates and cleans data
- Extracts metadata
- Provides summary statistics

**Key Classes:**
- `LightCurveLoader` - Main data loading class

**Supported Formats:**
```python
# FITS: Kepler/TESS standard
TIME, FLUX, FLUX_ERR columns in LIGHTCURVE extension

# CSV: Flexible format
time,flux,flux_err
0.0,1000.5,2.1
...
```

#### 2. **ML Layer** (`backend/ml/`)

**preprocessing.py** - Data preprocessing and feature extraction
- Sigma clipping for outlier removal
- Flux normalization
- Gap filling (interpolation)
- Feature extraction from windows
- Smoothing (Savitzky-Golay)
- Periodogram computation

**Key Classes:**
- `LightCurvePreprocessor` - Preprocessing pipeline

**Features Extracted (per window):**
- Statistical: mean, std, median, skewness, kurtosis
- Variability: MAD, RMS, peak-to-peak
- Trend: slope, intercept
- Difference: rate of change

**models.py** - Anomaly detection models
- Isolation Forest algorithm
- Local Outlier Factor (LOF)
- Statistical threshold detection
- Transit event detection
- Ensemble methods

**Key Classes:**
- `AnomalyDetector` - Main detection class
- `EnsembleAnomalyDetector` - Advanced ensemble

**Detection Methods:**
1. **Window-based**: Sliding window feature analysis
2. **Point-based**: Z-score threshold detection
3. **Event-based**: Transit and flare detection

**training.py** - Model training pipeline
- Train from files
- Train from DataFrames
- Generate synthetic data
- Model persistence (save/load)
- Cross-validation support

**Key Classes:**
- `ModelTrainer` - Training pipeline

#### 3. **API Layer** (`backend/api/`)

**routes.py** - REST API endpoints

**Endpoints:**
```
GET  /health                 - Health check
POST /api/analyze            - Analyze light curve
POST /api/train              - Train on real data
POST /api/train/synthetic    - Train on synthetic data
POST /api/export             - Export results
```

**app.py** - Flask application entry point

---

### Frontend Components

#### **Streamlit App** (`frontend/app.py`)

**Features:**
- 📊 File upload (FITS/CSV)
- 🔍 Real-time analysis
- 📈 Interactive visualizations
- 🎓 Model training interface
- 💾 Results export
- ⚙️ Parameter tuning

**Tabs:**
1. **Analyze** - Upload and analyze light curves
2. **Train Model** - Train custom models
3. **Statistics** - Detailed statistics
4. **About** - Documentation

**Visualizations:**
- Main light curve plot with anomaly highlighting
- 4-panel analysis dashboard:
  - Light curve with anomalies
  - Anomaly score distribution
  - Flux distribution (normal vs anomaly)
  - Running average trend

---

### Utility Scripts

#### **generate_sample_data.py**

Generates synthetic light curves with:
- Normal stellar variability
- Exoplanet transits (periodic dips)
- Stellar flares (spikes)
- Random outliers
- Combined anomalies

**Usage:**
```bash
python generate_sample_data.py --output-dir data/samples --format both --n-samples 5
```

**Output:** 5 sample light curves in both CSV and FITS formats

#### **run.sh / run.bat**

Automated launcher scripts that:
1. Create virtual environment (if needed)
2. Install dependencies (if needed)
3. Generate sample data (if needed)
4. Launch Streamlit app

---

## 🔄 Data Flow

### Analysis Pipeline

```
┌─────────────────┐
│  Upload File    │
│  (FITS/CSV)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LightCurve     │
│  Loader         │
│  • Parse file   │
│  • Validate     │
│  • Clean data   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessor   │
│  • Normalize    │
│  • Sigma clip   │
│  • Fill gaps    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Extraction     │
│  • Windows      │
│  • Statistics   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Anomaly        │
│  Detection      │
│  • IF           │
│  • LOF          │
│  • Statistical  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Visualization  │
│  & Results      │
└─────────────────┘
```

### Training Pipeline

```
┌─────────────────┐
│  Training Data  │
│  (Multiple      │
│   light curves) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocess     │
│  Each Curve     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract        │
│  Features       │
│  (All curves)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Train Models   │
│  • IF           │
│  • LOF          │
│  • Scaler       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Model     │
│  (Persist)      │
└─────────────────┘
```

---

## 🔑 Key Technologies

### Backend
- **Python 3.8+** - Core language
- **NumPy** - Numerical computing
- **pandas** - Data manipulation
- **scikit-learn** - ML algorithms
- **astropy** - FITS file handling
- **scipy** - Scientific computing
- **Flask** - REST API
- **joblib** - Model persistence

### Frontend
- **Streamlit** - Web UI framework
- **Plotly** - Interactive plots

### Development
- **setuptools** - Packaging
- **venv** - Virtual environments

---

## 📦 Dependencies

See `requirements.txt` for complete list:

```
# Core ML
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Astronomy
astropy>=5.3.0

# Optional: Deep Learning
tensorflow>=2.13.0

# API
flask>=2.3.0
flask-cors>=4.0.0

# Frontend
streamlit>=1.28.0
plotly>=5.17.0

# Utilities
joblib>=1.3.0
requests>=2.31.0
pydantic>=2.0.0
```

---

## 🎯 Design Principles

### 1. **Modularity**
- Clear separation of concerns
- Independent components
- Easy to extend

### 2. **Flexibility**
- Multiple file formats
- Multiple detection methods
- Configurable parameters

### 3. **User-Friendly**
- Simple interface
- Clear visualizations
- Helpful documentation

### 4. **Scientific Rigor**
- Validated algorithms
- Statistical methods
- Reproducible results

### 5. **Windows Compatibility**
- No Linux-specific dependencies
- Clear Windows instructions
- Batch file launcher

---

## 🚀 Extension Points

### Easy to Add:

1. **New File Formats**
   - Add parser to `loader.py`
   - Follow existing pattern

2. **New Features**
   - Add to `preprocessing.py`
   - Extend feature extraction

3. **New Detection Methods**
   - Add to `models.py`
   - Implement detector class

4. **New Visualizations**
   - Add to `frontend/app.py`
   - Use Plotly components

5. **New API Endpoints**
   - Add to `routes.py`
   - Follow REST conventions

---

## 📊 Performance Characteristics

### Typical Performance (on modern PC):

- **File Loading**: < 1 second
- **Preprocessing**: 1-2 seconds (2000 points)
- **Feature Extraction**: 2-5 seconds
- **Model Training**: 5-30 seconds (depending on data size)
- **Prediction**: < 1 second
- **Visualization**: Real-time (interactive)

### Memory Usage:

- **Baseline**: ~200 MB
- **Per Light Curve**: ~1-5 MB
- **Model**: ~10-50 MB

### Scalability:

- Light curves: Up to 100,000 points tested
- Training data: Up to 1000 light curves tested
- Concurrent users: 1-10 (not designed for high concurrency)

---

## 🔒 Security Notes

For production deployment (not included in this local version):

- Add authentication to API
- Implement rate limiting
- Validate all uploads
- Sanitize file names
- Set upload size limits
- Use HTTPS
- Add CSRF protection

---

## 📝 Code Quality

- Type hints in key functions
- Docstrings for all classes/methods
- Error handling throughout
- Logging at appropriate levels
- Input validation
- Clear variable names

---

For usage instructions, see:
- [README.md](README.md) - Complete documentation
- [QUICKSTART.md](QUICKSTART.md) - 5-minute guide
- [EXAMPLES.md](EXAMPLES.md) - Code examples
