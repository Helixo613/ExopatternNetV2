"""
Streamlit frontend for Stellar Light Curve Anomaly Detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import io

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.data import LightCurveLoader
from backend.ml import LightCurvePreprocessor, AnomalyDetector, ModelTrainer


# Page configuration
st.set_page_config(
    page_title="Stellar Anomaly Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = None


def load_model():
    """Load or create anomaly detection model."""
    try:
        trainer = ModelTrainer()
        detector = trainer.load_model('default_model')
        st.session_state.model_loaded = True
        return detector, trainer.preprocessor
    except:
        # Create new model with synthetic data
        with st.spinner('No pre-trained model found. Creating baseline model with synthetic data...'):
            trainer = ModelTrainer()
            detector = trainer.train_with_synthetic_data(n_samples=100, contamination=0.1)
            st.session_state.model_loaded = True
        return detector, trainer.preprocessor


def create_lightcurve_plot(df, anomaly_mask=None, title="Stellar Light Curve"):
    """Create interactive light curve plot with anomaly highlighting."""
    fig = go.Figure()

    # Plot normal points
    if anomaly_mask is not None:
        normal_mask = ~anomaly_mask
        fig.add_trace(go.Scatter(
            x=df['time'][normal_mask],
            y=df['flux'][normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='#1f77b4', size=4, opacity=0.6),
            hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<extra></extra>'
        ))

        # Plot anomalous points
        fig.add_trace(go.Scatter(
            x=df['time'][anomaly_mask],
            y=df['flux'][anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='#d62728', size=6, symbol='star'),
            hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<br><b>ANOMALY</b><extra></extra>'
        ))
    else:
        # Plot all points
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['flux'],
            mode='markers',
            name='Flux',
            marker=dict(color='#1f77b4', size=4, opacity=0.6),
            hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Flux',
        hovermode='closest',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )

    return fig


def create_analysis_dashboard(df, results):
    """Create comprehensive analysis dashboard."""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Light Curve with Anomalies',
            'Anomaly Score Distribution',
            'Flux Distribution',
            'Time Series Decomposition'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "histogram"}, {"type": "scatter"}]
        ]
    )

    anomaly_mask = np.array(results['anomaly_mask'])

    # 1. Light curve with anomalies
    fig.add_trace(
        go.Scatter(
            x=df['time'][~anomaly_mask],
            y=df['flux'][~anomaly_mask],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=3),
            showlegend=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['time'][anomaly_mask],
            y=df['flux'][anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=5, symbol='star'),
            showlegend=True
        ),
        row=1, col=1
    )

    # 2. Anomaly scores
    if 'anomaly_scores' in results:
        scores = np.array(results['anomaly_scores'])
        fig.add_trace(
            go.Histogram(
                x=scores,
                name='Anomaly Scores',
                marker_color='purple',
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. Flux distribution
    fig.add_trace(
        go.Histogram(
            x=df['flux'][~anomaly_mask],
            name='Normal Flux',
            marker_color='blue',
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=df['flux'][anomaly_mask],
            name='Anomalous Flux',
            marker_color='red',
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. Running average
    window_size = min(50, len(df) // 10)
    running_avg = df['flux'].rolling(window=window_size, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=df['time'],
            y=running_avg,
            mode='lines',
            name='Running Average',
            line=dict(color='green', width=2),
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white'
    )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Anomaly Score", row=1, col=2)
    fig.update_xaxes(title_text="Flux", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)

    fig.update_yaxes(title_text="Flux", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Flux", row=2, col=2)

    return fig


def analyze_lightcurve(uploaded_file, detector, preprocessor, contamination, window_size):
    """Analyze uploaded light curve."""
    try:
        # Load data
        loader = LightCurveLoader()

        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{uploaded_file.name}")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        df = loader.load_file(str(temp_path))

        # Get statistics
        stats = loader.get_summary_stats(df)

        # Preprocess
        df_processed = preprocessor.preprocess(df.copy(), normalize=True)

        # Extract features
        features = preprocessor.extract_features(df_processed, window_size)

        # Predict anomalies
        predictions, scores = detector.predict_with_scores(features)

        # Map window predictions to points
        stride = max(1, window_size // 4)
        anomaly_mask = np.zeros(len(df_processed), dtype=bool)

        for i, pred in enumerate(predictions):
            if pred == -1:
                start_idx = i * stride
                end_idx = min(start_idx + window_size, len(df_processed))
                anomaly_mask[start_idx:end_idx] = True

        # Point anomalies
        point_anomalies = detector.detect_point_anomalies(df_processed, threshold=3.0)

        # Transit events
        transit_events = detector.detect_transit_events(df_processed, depth_threshold=0.01, duration_min=3)

        # Clean up
        temp_path.unlink()

        return {
            'df': df_processed,
            'stats': stats,
            'anomaly_mask': anomaly_mask,
            'anomaly_scores': scores,
            'point_anomalies': point_anomalies,
            'transit_events': transit_events,
            'n_anomalies': int(np.sum(anomaly_mask)),
            'anomaly_rate': float(np.sum(anomaly_mask) / len(anomaly_mask))
        }

    except Exception as e:
        st.error(f"Error analyzing light curve: {str(e)}")
        return None


# Main application
def main():
    # Header
    st.markdown('<p class="main-header"> Stellar Light Curve Anomaly Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced ML-based detection of exoplanet transits, stellar flares, and anomalous patterns</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.title(" Configuration")

    # Model status
    st.sidebar.subheader("Model Status")
    if st.session_state.model_loaded:
        st.sidebar.success(" Model Loaded")
    else:
        st.sidebar.warning(" Model Not Loaded")

    # Load model button
    if st.sidebar.button(" Initialize/Reload Model"):
        with st.spinner("Loading model..."):
            detector, preprocessor = load_model()
            st.session_state.detector = detector
            st.session_state.preprocessor = preprocessor
        st.sidebar.success("Model loaded successfully!")

    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    contamination = st.sidebar.slider(
        "Expected Anomaly Rate",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Expected proportion of anomalies in the data"
    )

    window_size = st.sidebar.slider(
        "Analysis Window Size",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of points in each analysis window"
    )

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([" Analyze", " Train Model", " Statistics", "ℹ About"])

    with tab1:
        st.header("Upload and Analyze Light Curve")

        uploaded_file = st.file_uploader(
            "Choose a light curve file (FITS or CSV)",
            type=['fits', 'fit', 'csv'],
            help="Upload a stellar light curve file in FITS or CSV format"
        )

        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")

            if st.button(" Analyze Light Curve", type="primary"):
                # Ensure model is loaded
                if not st.session_state.model_loaded:
                    detector, preprocessor = load_model()
                    st.session_state.detector = detector
                    st.session_state.preprocessor = preprocessor

                with st.spinner("Analyzing light curve..."):
                    results = analyze_lightcurve(
                        uploaded_file,
                        st.session_state.detector,
                        st.session_state.preprocessor,
                        contamination,
                        window_size
                    )

                if results:
                    st.session_state.analysis_done = True
                    st.session_state.results = results

            # Display results
            if st.session_state.analysis_done and st.session_state.results:
                results = st.session_state.results
                df = results['df']

                st.success("✅ Analysis Complete!")

                # Summary metrics
                st.subheader(" Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Points", results['stats']['n_points'])
                with col2:
                    st.metric("Anomalies Detected", results['n_anomalies'])
                with col3:
                    st.metric("Anomaly Rate", f"{results['anomaly_rate']*100:.2f}%")
                with col4:
                    st.metric("Transit Events", len(results['transit_events']))

                # Main plot
                st.subheader(" Light Curve Visualization")
                fig = create_lightcurve_plot(df, results['anomaly_mask'])
                st.plotly_chart(fig, use_container_width=True)

                # Detailed dashboard
                st.subheader(" Detailed Analysis Dashboard")
                dashboard_fig = create_analysis_dashboard(df, results)
                st.plotly_chart(dashboard_fig, use_container_width=True)

                # Anomaly details
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(" Point Anomalies")
                    point_anom = results['point_anomalies']
                    st.write(f"**Total Point Anomalies:** {point_anom['n_anomalies']}")
                    st.write(f"**Flux Dips:** {point_anom['n_dips']}")
                    st.write(f"**Flux Spikes:** {point_anom['n_spikes']}")

                with col2:
                    st.subheader(" Transit Events")
                    if results['transit_events']:
                        for i, event in enumerate(results['transit_events'][:5]):  # Show first 5
                            with st.expander(f"Event {i+1}"):
                                st.write(f"**Time:** {event['start_time']:.2f} - {event['end_time']:.2f}")
                                st.write(f"**Duration:** {event['time_duration']:.2f} time units")
                                st.write(f"**Depth:** {event['depth']*100:.2f}%")
                    else:
                        st.info("No transit events detected")

                # Export results
                st.subheader(" Export Results")
                export_df = df.copy()
                export_df['is_anomaly'] = results['anomaly_mask']

                csv = export_df.to_csv(index=False)
                st.download_button(
                    label=" Download Results (CSV)",
                    data=csv,
                    file_name=f"anomaly_results_{uploaded_file.name}.csv",
                    mime="text/csv"
                )

    with tab2:
        st.header("Train Custom Model")
        st.write("Train a new anomaly detection model on your own data or synthetic data.")

        train_option = st.radio(
            "Training Option",
            ["Synthetic Data", "Upload Training Files"]
        )

        if train_option == "Synthetic Data":
            st.subheader("Generate Synthetic Training Data")

            n_samples = st.number_input(
                "Number of synthetic light curves",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )

            train_contamination = st.slider(
                "Contamination (anomaly rate)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01
            )

            if st.button("🎓 Train on Synthetic Data"):
                with st.spinner(f"Training model on {n_samples} synthetic light curves..."):
                    trainer = ModelTrainer()
                    detector = trainer.train_with_synthetic_data(n_samples, train_contamination)
                    st.session_state.detector = detector
                    st.session_state.preprocessor = trainer.preprocessor
                    st.session_state.model_loaded = True

                st.success(" Model training complete!")
                st.balloons()

        else:
            st.subheader("Upload Training Files")
            training_files = st.file_uploader(
                "Upload multiple light curve files",
                type=['fits', 'fit', 'csv'],
                accept_multiple_files=True
            )

            if training_files and st.button("🎓 Train Model"):
                with st.spinner(f"Training on {len(training_files)} files..."):
                    # Save files temporarily
                    temp_paths = []
                    for file in training_files:
                        temp_path = Path(f"/tmp/{file.name}")
                        with open(temp_path, 'wb') as f:
                            f.write(file.getbuffer())
                        temp_paths.append(str(temp_path))

                    # Train
                    trainer = ModelTrainer()
                    detector = trainer.train_from_files(temp_paths, contamination, window_size)

                    # Clean up
                    for temp_path in temp_paths:
                        Path(temp_path).unlink()

                    st.session_state.detector = detector
                    st.session_state.preprocessor = trainer.preprocessor
                    st.session_state.model_loaded = True

                st.success(" Model training complete!")
                st.balloons()

    with tab3:
        st.header("Statistical Information")

        if st.session_state.analysis_done and st.session_state.results:
            results = st.session_state.results
            stats = results['stats']

            st.subheader(" Light Curve Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Time Information**")
                st.write(f"- Start Time: {stats['time_start']:.2f}")
                st.write(f"- End Time: {stats['time_end']:.2f}")
                st.write(f"- Time Span: {stats['time_span']:.2f}")
                st.write(f"- Number of Points: {stats['n_points']}")

            with col2:
                st.write("**Flux Statistics**")
                st.write(f"- Mean: {stats['flux_mean']:.2f}")
                st.write(f"- Median: {stats['flux_median']:.2f}")
                st.write(f"- Std Dev: {stats['flux_std']:.2f}")
                st.write(f"- Min: {stats['flux_min']:.2f}")
                st.write(f"- Max: {stats['flux_max']:.2f}")

        else:
            st.info(" Upload and analyze a light curve to see statistics")

    with tab4:
        st.header("About This Application")

        st.markdown("""
        ###  Stellar Light Curve Anomaly Detector

        This application uses advanced machine learning techniques to detect anomalies in stellar light curves,
        including:

        - **Exoplanet Transits**: Periodic dips in brightness caused by planets passing in front of stars
        - **Stellar Flares**: Sudden brightness increases from stellar activity
        - **Noise Artifacts**: Instrumental or environmental anomalies
        - **Other Unusual Patterns**: Any deviation from normal stellar behavior

        ###  Machine Learning Approach

        The system uses an ensemble of anomaly detection algorithms:

        1. **Isolation Forest**: Tree-based anomaly detection
        2. **Local Outlier Factor**: Density-based anomaly detection
        3. **Statistical Methods**: Z-score and threshold-based detection

        ###  Features

        - **Multi-format Support**: FITS and CSV files
        - **Interactive Visualization**: Zoom, pan, and explore your data
        - **Detailed Analysis**: Multiple detection methods and statistics
        - **Model Training**: Train on your own data or synthetic data
        - **Export Results**: Download analysis results as CSV

        ###  Getting Started

        1. **Initialize Model**: Click "Initialize/Reload Model" in the sidebar
        2. **Upload Data**: Upload a light curve file (FITS or CSV)
        3. **Analyze**: Click "Analyze Light Curve" to detect anomalies
        4. **Explore**: View visualizations and export results

        ###  References

        - Based on research from astronomical anomaly detection papers
        - Inspired by WaldoInSky and StellarScope projects
        - Uses scikit-learn, astropy, and plotly libraries

        ###  Tips

        - Adjust the "Expected Anomaly Rate" to tune sensitivity
        - Use larger window sizes for detecting longer-duration events
        - Train custom models on your specific data for best results

        ---

        **Version**: 1.0.0 | **Built with**: Streamlit, scikit-learn, astropy, plotly
        """)


if __name__ == '__main__':
    main()
