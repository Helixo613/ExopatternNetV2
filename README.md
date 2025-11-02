# 🌟 Stellar Light Curve Anomaly Detector

A complete machine learning system for detecting anomalies in stellar light curves, including exoplanet transits, stellar flares, and noise artifacts. Built with Python, scikit-learn, and Streamlit for a professional, interactive user experience.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **🔬 Advanced ML Detection**: Ensemble of Isolation Forest, Local Outlier Factor, and statistical methods
- **📊 Interactive Visualization**: Real-time, zoomable plots with anomaly highlighting using Plotly
- **📁 Multi-format Support**: FITS and CSV file formats
- **🎓 Model Training**: Train on your own data or synthetic light curves
- **💾 Export Results**: Download analysis results as CSV
- **🖥️ Professional UI**: Clean Streamlit interface with comprehensive dashboards
- **🪟 Windows Compatible**: Runs smoothly on Windows (and Linux/Mac)

## 🎯 What It Detects

1. **Exoplanet Transits** - Periodic dips in brightness from planets passing in front of stars
2. **Stellar Flares** - Sudden brightness increases from stellar activity
3. **Instrumental Artifacts** - Noise and outliers from measurement errors
4. **Other Anomalies** - Any unusual patterns in stellar light curves

## 🏗️ Architecture

```
ExopatternNetV3/
├── backend/                 # ML Backend Service
│   ├── api/                # Flask API routes
│   ├── ml/                 # ML models and training
│   ├── data/               # Data loading and processing
│   └── models/             # Saved trained models
├── frontend/               # Streamlit UI
├── data/samples/           # Sample light curves
├── notebooks/              # Jupyter notebooks (optional)
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Windows 10/11, Linux, or macOS

### Installation

1. **Clone or download this repository**

```bash
cd ExopatternNetV3
```

2. **Create a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Generate sample data (optional)**

```bash
python generate_sample_data.py --format both --n-samples 5
```

This creates sample light curves in `data/samples/` directory with various anomaly types.

### Running the Application

#### Option 1: Streamlit Frontend (Recommended)

The Streamlit app provides a complete, user-friendly interface with built-in ML processing:

```bash
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`

#### Option 2: Backend API + Frontend (Advanced)

For more control, run the Flask backend separately:

**Terminal 1 - Start Backend API:**
```bash
python backend/app.py
```

The API will start on `http://localhost:5000`

**Terminal 2 - Start Frontend:**
```bash
streamlit run frontend/app.py
```

The frontend will connect to the backend API.

## 📖 Usage Guide

### 1. Analyzing a Light Curve

1. **Launch the application**:
   ```bash
   streamlit run frontend/app.py
   ```

2. **Initialize the model**: Click "Initialize/Reload Model" in the sidebar

3. **Upload your data**:
   - Go to the "📊 Analyze" tab
   - Click "Choose a light curve file"
   - Select a FITS or CSV file

4. **Analyze**:
   - Click "🔍 Analyze Light Curve"
   - Wait for processing to complete

5. **Explore results**:
   - View the interactive light curve plot with anomalies highlighted in red
   - Check the detailed analysis dashboard
   - Review statistics and detected events
   - Download results as CSV

### 2. Training a Custom Model

#### Using Synthetic Data (Quick Start)

1. Go to the "🎓 Train Model" tab
2. Select "Synthetic Data"
3. Choose the number of synthetic light curves (100 recommended)
4. Set contamination rate (0.1 = 10% anomalies)
5. Click "🎓 Train on Synthetic Data"

#### Using Your Own Data

1. Go to the "🎓 Train Model" tab
2. Select "Upload Training Files"
3. Upload multiple light curve files
4. Click "🎓 Train Model"
5. The model will be trained and automatically loaded

### 3. Adjusting Detection Parameters

In the sidebar, you can adjust:

- **Expected Anomaly Rate**: Higher values make the detector more sensitive
- **Analysis Window Size**: Larger windows detect longer-duration events

### 4. File Format Requirements

#### CSV Format
```csv
time,flux,flux_err
0.0,1000.5,2.3
0.1,1002.1,2.1
...
```

Required columns: `time`, `flux`
Optional: `flux_err` (measurement uncertainty)

#### FITS Format

Standard Kepler/TESS light curve format:
- Extension: `LIGHTCURVE` (or first binary table)
- Columns: `TIME`, `FLUX` (or `SAP_FLUX`, `PDCSAP_FLUX`)
- Optional: `FLUX_ERR`

## 🔬 Machine Learning Approach

### Algorithms

1. **Isolation Forest**
   - Tree-based anomaly detection
   - Isolates anomalies by random partitioning
   - Effective for global anomalies

2. **Local Outlier Factor (LOF)**
   - Density-based detection
   - Identifies local outliers in feature space
   - Good for contextual anomalies

3. **Statistical Methods**
   - Z-score thresholding
   - Transit detection (sustained dips)
   - Flare detection (sudden spikes)

### Feature Extraction

For each analysis window, we extract:

- **Statistical features**: mean, std, median, skewness, kurtosis
- **Variability metrics**: MAD, RMS, peak-to-peak
- **Trend features**: slope, intercept from linear fit
- **Difference features**: rate of change
- **Percentiles**: 5th, 25th, 75th, 95th

### Preprocessing Pipeline

1. **Sigma clipping**: Remove extreme outliers (>5σ)
2. **Normalization**: Zero mean, unit variance
3. **Gap filling**: Linear interpolation for small gaps
4. **Smoothing**: Savitzky-Golay filter (optional)

## 📊 Sample Data

The `generate_sample_data.py` script creates five types of light curves:

1. **normal_star**: Clean light curve with only stellar variability
2. **exoplanet_transit**: Periodic transits every 15 days
3. **stellar_flares**: Multiple flare events
4. **noisy_outliers**: Random instrumental artifacts
5. **complex_system**: Combination of all anomaly types

Generate samples:
```bash
python generate_sample_data.py --output-dir data/samples --format both --n-samples 5
```

## 🔧 Advanced Usage

### Using the Backend API

The Flask backend provides REST API endpoints:

#### Analyze Light Curve
```bash
curl -X POST http://localhost:5000/api/analyze \
  -F "file=@path/to/lightcurve.csv" \
  -F "contamination=0.1" \
  -F "method=ensemble"
```

#### Train Model
```bash
curl -X POST http://localhost:5000/api/train/synthetic \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 100, "contamination": 0.1}'
```

#### Health Check
```bash
curl http://localhost:5000/health
```

### Python API

You can also use the backend modules directly in Python:

```python
from backend.data import LightCurveLoader
from backend.ml import LightCurvePreprocessor, AnomalyDetector, ModelTrainer

# Load data
loader = LightCurveLoader()
df = loader.load_file('data/samples/exoplanet_transit.csv')

# Preprocess
preprocessor = LightCurvePreprocessor()
df_processed = preprocessor.preprocess(df, normalize=True)

# Extract features
features = preprocessor.extract_features(df_processed, window_size=50)

# Train detector
detector = AnomalyDetector(contamination=0.1)
detector.fit(features)

# Predict anomalies
predictions, scores = detector.predict_with_scores(features)

# Save model
detector.save_model('backend/models/my_model')
```

## 🧪 Testing

Generate and test with sample data:

```bash
# Generate sample data
python generate_sample_data.py

# Run the app
streamlit run frontend/app.py

# Upload one of the generated files from data/samples/
# Try: exoplanet_transit.csv or stellar_flares.csv
```

## 🛠️ Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the project root:
```bash
cd ExopatternNetV3
streamlit run frontend/app.py
```

### Model Not Loading

If the model fails to load, train a new one:
1. Go to "🎓 Train Model" tab
2. Click "Train on Synthetic Data"
3. Wait for training to complete

### FITS File Errors

Ensure you have astropy installed:
```bash
pip install astropy
```

### Windows-Specific Issues

If you encounter path issues on Windows, make sure to:
- Use `python` instead of `python3`
- Use `\` or `/` for paths
- Run commands from Command Prompt or PowerShell, not Git Bash

## 📚 References

### Inspiration & Research

- **WaldoInSky**: ["Where is Waldo (and his friends)?"](https://github.com/kushaltirumala/WaldoInSky) - Comparison of anomaly detection algorithms for time-domain astronomy
- **StellarScope**: [Kepler Lightcurve Analysis](https://github.com/Fastian-afk/Stellar-Scope) - Gradio-based light curve analyzer

### Libraries Used

- **scikit-learn**: Machine learning algorithms
- **astropy**: Astronomy data formats (FITS)
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations
- **streamlit**: Web interface
- **flask**: Backend API

## 🎓 Educational Resources

- [Kepler Mission - NASA](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [TESS Mission - NASA](https://tess.mit.edu/)
- [Exoplanet Detection Methods](https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/)
- [Anomaly Detection in Time Series](https://scikit-learn.org/stable/modules/outlier_detection.html)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional anomaly detection algorithms
- Deep learning models (LSTM, Transformer)
- Real-time streaming analysis
- Multi-light curve batch processing
- Advanced visualization options

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- NASA Kepler and TESS missions for light curve data formats
- scikit-learn community for ML algorithms
- Streamlit team for the amazing framework
- Astronomy community for anomaly detection research

## 📧 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the sample data to understand expected formats
3. Ensure all dependencies are installed correctly

---

**Built with ❤️ for the astronomy and ML communities**

*Detecting the unexpected in stellar light curves, one photon at a time.* ✨
