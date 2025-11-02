# 🚀 Quick Start Guide

Get up and running with the Stellar Light Curve Anomaly Detector in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd ExopatternNetV3

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Step 2: Generate Sample Data (1 minute)

```bash
python generate_sample_data.py
```

This creates 5 sample light curves in `data/samples/`:
- `normal_star.csv/fits` - Clean stellar data
- `exoplanet_transit.csv/fits` - Planet transits
- `stellar_flares.csv/fits` - Flare events
- `noisy_outliers.csv/fits` - Artifacts
- `complex_system.csv/fits` - Multiple anomalies

## Step 3: Launch the Application (1 minute)

```bash
streamlit run frontend/app.py
```

Your browser will open automatically to `http://localhost:8501`

## Step 4: Analyze Your First Light Curve (1 minute)

1. **Initialize Model**: Click "🔄 Initialize/Reload Model" in the left sidebar
   - Wait ~10 seconds for synthetic training data generation

2. **Upload Data**:
   - Go to "📊 Analyze" tab
   - Click "Browse files"
   - Select `data/samples/exoplanet_transit.csv`

3. **Analyze**:
   - Click "🔍 Analyze Light Curve"
   - Wait for processing (~5-10 seconds)

4. **View Results**:
   - See anomalies highlighted in red on the plot
   - Check the "Transit Events" section - you should see periodic transits detected!
   - Scroll down to see detailed analysis dashboard

## Step 5: Experiment! (<1 minute)

Try different files:
- `stellar_flares.csv` - See spike detection
- `complex_system.csv` - Multiple anomaly types
- `normal_star.csv` - Should detect very few anomalies

Adjust parameters in sidebar:
- Increase "Expected Anomaly Rate" → More sensitive
- Change "Analysis Window Size" → Detect different duration events

## Next Steps

### Train on Your Own Data

1. Go to "🎓 Train Model" tab
2. Upload your FITS or CSV files
3. Click "🎓 Train Model"

### Export Results

After analysis:
1. Scroll to bottom of results
2. Click "📥 Download Results (CSV)"
3. Open in Excel, Python, or any data tool

### Use Your Own Light Curves

Your files must have:
- **CSV**: Columns named `time`, `flux` (optional: `flux_err`)
- **FITS**: Standard Kepler/TESS format with LIGHTCURVE extension

## Troubleshooting

**"Model not loaded" error?**
- Click "🔄 Initialize/Reload Model" in sidebar
- Or go to "🎓 Train Model" → "Train on Synthetic Data"

**Import errors?**
```bash
pip install -r requirements.txt --upgrade
```

**Can't find sample data?**
```bash
python generate_sample_data.py
```

## Visual Guide

```
┌─────────────────────────────────────┐
│  Sidebar                            │
│  ⚙️ Configuration                   │
│  ┌──────────────────────┐          │
│  │ 🔄 Initialize Model  │          │
│  └──────────────────────┘          │
│                                     │
│  Expected Anomaly Rate: [====○]    │
│  Analysis Window Size:  [====○]    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Main Panel                         │
│  📊 Analyze | 🎓 Train | 📈 Stats  │
│                                     │
│  Upload: [Browse files...]          │
│  ┌──────────────────────┐          │
│  │ 🔍 Analyze Light     │          │
│  │    Curve             │          │
│  └──────────────────────┘          │
│                                     │
│  📊 Results:                        │
│  [Interactive Plot with Anomalies]  │
│  🔴 Anomalies: 234                  │
│  🌑 Transits: 5                     │
│                                     │
│  💾 [Download CSV]                  │
└─────────────────────────────────────┘
```

## What to Expect

### Exoplanet Transit Detection
- Periodic dips every ~15 days
- Detected as "Transit Events"
- Red markers on plot

### Stellar Flare Detection
- Sudden spikes in flux
- Detected as "Spike" anomalies
- Check "Point Anomalies" section

### Normal Stars
- Few anomalies (~1-5%)
- Mostly stellar variability
- Should be mostly blue points

## Tips for Best Results

1. **Clean Data**: Remove known bad sections before upload
2. **Adjust Sensitivity**: Start with 10% anomaly rate, adjust as needed
3. **Window Size**:
   - Small (10-30): Detect short events (flares)
   - Medium (50-100): Detect transits
   - Large (100-200): Detect long-term trends
4. **Train Custom Model**: Use similar stars for training

---

**You're ready to discover anomalies in stellar light curves!** 🌟

For more details, see [README.md](README.md)
