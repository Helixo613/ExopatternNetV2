"""
Streamlit frontend for Stellar Light Curve Anomaly Detection.
Research-grade UI with dark science theme.
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
from backend.ml.models import map_window_predictions_to_points
from backend.ml.model_registry import list_models, get_model, get_display_name, load_pretrained, MODEL_DISPLAY_NAMES, CLASSICAL_MODELS, DEEP_MODELS
from backend.ml.evaluation import AnomalyEvaluator
from backend.ml.feature_names import FEATURE_GROUPS, get_feature_names

# Path to pre-trained model artifacts
ARTIFACTS_DIR = str(Path(__file__).parent.parent / 'artifacts' / 'models')

# --- Design system constants ---
COLORS = {
    'bg_deep': '#0d1117',
    'bg_surface': '#161b22',
    'bg_raised': '#21262d',
    'border': '#30363d',
    'accent_blue': '#58a6ff',
    'accent_green': '#3fb950',
    'accent_amber': '#d29922',
    'accent_red': '#f85149',
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'purple': '#9e6bdb',
}

# Page configuration
st.set_page_config(
    page_title="ExoPattern -- Stellar Anomaly Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': 'ExoPattern v3.0 -- Stellar Light Curve Anomaly Detection System'
    }
)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Container padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
.exo-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0.15rem;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.exo-header .version-badge {
    font-size: 0.7rem;
    font-weight: 600;
    background: #21262d;
    color: #58a6ff;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    border: 1px solid #30363d;
    vertical-align: middle;
    position: relative;
    top: -2px;
}
.exo-tagline {
    font-size: 0.95rem;
    color: #8b949e;
    margin-bottom: 0.5rem;
    font-weight: 400;
}
.exo-rule {
    border: none;
    border-top: 1px solid #30363d;
    margin: 0.8rem 0 1.2rem 0;
}

/* Sidebar section titles */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #30363d;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
}

/* Status badge pills */
.status-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.2rem 0.65rem;
    border-radius: 12px;
    margin: 0.3rem 0;
}
.status-badge.ready {
    background: rgba(63, 185, 80, 0.15);
    color: #3fb950;
    border: 1px solid rgba(63, 185, 80, 0.3);
}
.status-badge.pending {
    background: rgba(210, 153, 34, 0.15);
    color: #d29922;
    border: 1px solid rgba(210, 153, 34, 0.3);
}
.status-badge.model-type {
    background: rgba(88, 166, 255, 0.12);
    color: #58a6ff;
    border: 1px solid rgba(88, 166, 255, 0.25);
    font-size: 0.7rem;
}

/* Metric cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.5rem;
}
.metric-card.blue { border-left: 3px solid #58a6ff; }
.metric-card.green { border-left: 3px solid #3fb950; }
.metric-card.amber { border-left: 3px solid #d29922; }
.metric-card.red { border-left: 3px solid #f85149; }
.metric-card.purple { border-left: 3px solid #9e6bdb; }

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8b949e;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1.2;
}
.metric-subtext {
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 0.15rem;
}

/* Tab overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #30363d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: #8b949e;
    padding: 0.6rem 1.2rem;
    border-bottom: 2px solid transparent;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3;
    border-bottom: 2px solid #58a6ff;
    background: transparent;
}

/* Section headers */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #e6edf3;
    padding-left: 0.75rem;
    border-left: 3px solid #58a6ff;
    margin: 1.5rem 0 0.8rem 0;
}

/* Expander overrides */
.streamlit-expanderHeader {
    font-size: 0.85rem;
    font-weight: 500;
    color: #e6edf3;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
}

/* Alert overrides */
div[data-testid="stAlert"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    color: #e6edf3;
}

/* Custom success banner */
.success-banner {
    background: rgba(63, 185, 80, 0.1);
    border: 1px solid rgba(63, 185, 80, 0.25);
    border-radius: 6px;
    padding: 0.6rem 1rem;
    color: #3fb950;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0.8rem 0;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #8b949e;
}
.empty-state .icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    opacity: 0.5;
}
.empty-state .message {
    font-size: 0.9rem;
}

/* Eval metrics panel */
.eval-panel {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.2rem;
    margin: 1rem 0;
}
.eval-panel .panel-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8b949e;
    margin-bottom: 0.8rem;
}
.eval-metric-row {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.eval-metric {
    text-align: center;
    min-width: 80px;
}
.eval-metric .label {
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.eval-metric .value {
    font-size: 1.3rem;
    font-weight: 700;
    line-height: 1.4;
}
.eval-metric .value.good { color: #3fb950; }
.eval-metric .value.ok { color: #d29922; }
.eval-metric .value.poor { color: #f85149; }
.eval-metric .value.na { color: #8b949e; }

/* Styled data tables */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}
.data-table th {
    text-align: left;
    font-weight: 600;
    color: #8b949e;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #30363d;
}
.data-table td {
    padding: 0.45rem 0.8rem;
    color: #e6edf3;
    border-bottom: 1px solid rgba(48, 54, 61, 0.5);
}
.data-table tr:hover td {
    background: rgba(88, 166, 255, 0.04);
}

/* Callout box */
.callout-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #e6edf3;
}

/* Sidebar footer */
.sidebar-footer {
    font-size: 0.7rem;
    color: #484f58;
    border-top: 1px solid #21262d;
    padding-top: 0.8rem;
    margin-top: 1.5rem;
    line-height: 1.5;
}

/* Hide hamburger menu icon for cleaner look */
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Chart theme helper ---
def apply_chart_theme(fig, height=500):
    """Apply unified dark science theme to a plotly figure."""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor=COLORS['bg_deep'],
        plot_bgcolor=COLORS['bg_deep'],
        font=dict(family='Inter, sans-serif', color=COLORS['text_primary'], size=12),
        title_font=dict(size=14, color=COLORS['text_primary']),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(size=11, color=COLORS['text_secondary']),
        ),
        height=height,
        margin=dict(l=50, r=20, t=50, b=50),
        hovermode='closest',
    )
    fig.update_xaxes(
        gridcolor='rgba(48,54,61,0.5)',
        zerolinecolor='#30363d',
        title_font=dict(size=12, color=COLORS['text_secondary']),
        tickfont=dict(size=10, color=COLORS['text_secondary']),
    )
    fig.update_yaxes(
        gridcolor='rgba(48,54,61,0.5)',
        zerolinecolor='#30363d',
        title_font=dict(size=12, color=COLORS['text_secondary']),
        tickfont=dict(size=10, color=COLORS['text_secondary']),
    )
    return fig


def metric_card(label, value, subtext="", accent="blue"):
    """Render a styled metric card."""
    return f"""
    <div class="metric-card {accent}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {'<div class="metric-subtext">' + subtext + '</div>' if subtext else ''}
    </div>
    """


def eval_score_class(val):
    """Return CSS class based on score quality."""
    if val is None or np.isnan(val):
        return 'na'
    if val >= 0.7:
        return 'good'
    if val >= 0.4:
        return 'ok'
    return 'poor'


def section_header(text):
    """Render a section header with left accent bar."""
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


def empty_state(message):
    """Render a styled empty state."""
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">&#9734;</div>
        <div class="message">{message}</div>
    </div>
    """, unsafe_allow_html=True)


# --- Initialize session state ---
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
        with st.spinner('No pre-trained model found. Creating baseline model with synthetic data...'):
            trainer = ModelTrainer()
            detector = trainer.train_with_synthetic_data(n_samples=100, contamination=0.1)
            st.session_state.model_loaded = True
        return detector, trainer.preprocessor


def create_lightcurve_plot(df, anomaly_mask=None, title="Stellar Light Curve"):
    """Create interactive light curve plot with anomaly highlighting."""
    fig = go.Figure()

    if anomaly_mask is not None:
        normal_mask = ~anomaly_mask
        fig.add_trace(go.Scatter(
            x=df['time'][normal_mask],
            y=df['flux'][normal_mask],
            mode='markers',
            name='Normal',
            marker=dict(color=COLORS['accent_blue'], size=4, opacity=0.6),
            hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=df['time'][anomaly_mask],
            y=df['flux'][anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color=COLORS['accent_red'], size=6, symbol='circle'),
            hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<br><b>ANOMALY</b><extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['flux'],
            mode='markers',
            name='Flux',
            marker=dict(color=COLORS['accent_blue'], size=4, opacity=0.6),
            hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Flux',
        showlegend=True,
    )
    apply_chart_theme(fig, height=500)

    return fig


def create_analysis_dashboard(df, results):
    """Create comprehensive analysis dashboard."""
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
            x=df['time'][~anomaly_mask], y=df['flux'][~anomaly_mask],
            mode='markers', name='Normal',
            marker=dict(color=COLORS['accent_blue'], size=3),
            showlegend=True
        ), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df['time'][anomaly_mask], y=df['flux'][anomaly_mask],
            mode='markers', name='Anomaly',
            marker=dict(color=COLORS['accent_red'], size=5, symbol='circle'),
            showlegend=True
        ), row=1, col=1
    )

    # 2. Anomaly scores
    if 'anomaly_scores' in results:
        scores = np.array(results['anomaly_scores'])
        fig.add_trace(
            go.Histogram(
                x=scores, name='Anomaly Scores',
                marker_color=COLORS['purple'], showlegend=False
            ), row=1, col=2
        )

    # 3. Flux distribution
    fig.add_trace(
        go.Histogram(
            x=df['flux'][~anomaly_mask], name='Normal Flux',
            marker_color=COLORS['accent_blue'], opacity=0.7, showlegend=False
        ), row=2, col=1
    )
    fig.add_trace(
        go.Histogram(
            x=df['flux'][anomaly_mask], name='Anomalous Flux',
            marker_color=COLORS['accent_red'], opacity=0.7, showlegend=False
        ), row=2, col=1
    )

    # 4. Running average
    window_size = min(50, len(df) // 10)
    running_avg = df['flux'].rolling(window=window_size, center=True).mean()
    fig.add_trace(
        go.Scatter(
            x=df['time'], y=running_avg,
            mode='lines', name='Running Average',
            line=dict(color=COLORS['accent_green'], width=2),
            showlegend=False
        ), row=2, col=2
    )

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Anomaly Score", row=1, col=2)
    fig.update_xaxes(title_text="Flux", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Flux", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Flux", row=2, col=2)

    fig.update_layout(showlegend=True)
    apply_chart_theme(fig, height=800)

    return fig


def analyze_lightcurve(uploaded_file, detector, preprocessor, contamination, window_size,
                       model_name='ensemble', feature_groups=None):
    """Analyze uploaded light curve."""
    try:
        loader = LightCurveLoader()

        temp_path = Path(f"/tmp/{uploaded_file.name}")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        df = loader.load_file(str(temp_path))
        stats = loader.get_summary_stats(df)

        has_labels = 'label' in df.columns
        true_labels = df['label'].values if has_labels else None

        df_processed = preprocessor.preprocess(df.copy(), normalize=True)
        features = preprocessor.extract_features(df_processed, window_size, feature_groups)

        if model_name != 'legacy' and model_name is not None:
            pretrained = load_pretrained(model_name, ARTIFACTS_DIR)
            if pretrained is not None:
                predictions = pretrained.predict(features)
                scores = pretrained.score_samples(features) if hasattr(pretrained, 'score_samples') else np.zeros(len(features))
            else:
                model = get_model(model_name, contamination=contamination)
                model.fit(features)
                predictions = model.predict(features)
                scores = model.score_samples(features) if hasattr(model, 'score_samples') else np.zeros(len(features))
        else:
            predictions, scores = detector.predict_with_scores(features)

        stride = max(1, window_size // 4)
        anomaly_mask, point_scores = map_window_predictions_to_points(
            predictions, scores, len(df_processed), window_size, stride, method='vote'
        )

        point_anomalies = detector.detect_point_anomalies(df_processed, threshold=3.0)
        transit_events = detector.detect_transit_events(df_processed, depth_threshold=0.01, duration_min=3)

        eval_metrics = None
        if has_labels and true_labels is not None:
            evaluator = AnomalyEvaluator()
            y_pred = anomaly_mask.astype(int)[:len(true_labels)]
            y_true = true_labels[:len(y_pred)]
            y_scores = -point_scores[:len(y_pred)]
            eval_metrics = evaluator.compute_metrics(y_true, y_pred, y_scores)

        temp_path.unlink()

        return {
            'df': df_processed,
            'stats': stats,
            'anomaly_mask': anomaly_mask,
            'anomaly_scores': scores,
            'point_anomalies': point_anomalies,
            'transit_events': transit_events,
            'n_anomalies': int(np.sum(anomaly_mask)),
            'anomaly_rate': float(np.sum(anomaly_mask) / len(anomaly_mask)),
            'eval_metrics': eval_metrics,
            'has_labels': has_labels,
        }

    except Exception as e:
        st.error(f"Error analyzing light curve: {str(e)}")
        return None


# --- Main application ---
def main():
    # Header
    st.markdown("""
    <div class="exo-header">ExoPattern <span class="version-badge">v3.0</span></div>
    <div class="exo-tagline">Stellar light curve anomaly detection across 8 models and 38 engineered features</div>
    <hr class="exo-rule">
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.markdown('<div class="sidebar-section">System Status</div>', unsafe_allow_html=True)

    if st.session_state.model_loaded:
        st.sidebar.markdown('<span class="status-badge ready">Model Ready</span>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<span class="status-badge pending">Model Not Loaded</span>', unsafe_allow_html=True)

    if st.sidebar.button("Initialize / Reload Model"):
        with st.spinner("Loading model..."):
            detector, preprocessor = load_model()
            st.session_state.detector = detector
            st.session_state.preprocessor = preprocessor

    # Model selection
    st.sidebar.markdown('<div class="sidebar-section">Model Selection</div>', unsafe_allow_html=True)
    available_models = list_models(include_deep=True)
    model_display = {name: get_display_name(name) for name in available_models}
    selected_model = st.sidebar.selectbox(
        "Detection Model",
        options=available_models,
        format_func=lambda x: model_display[x],
        index=available_models.index('ensemble') if 'ensemble' in available_models else 0,
        help="Choose the anomaly detection algorithm",
        label_visibility="collapsed",
    )

    # Model type indicator
    deep_models = {'cnn_autoencoder', 'lstm_autoencoder'}
    model_type = "Deep Learning" if selected_model in deep_models else "Classical"
    st.sidebar.markdown(f'<span class="status-badge model-type">{model_type}</span>', unsafe_allow_html=True)

    # Fixed parameters matching pre-trained models
    contamination = 0.1
    window_size = 50
    feature_groups = None  # all groups

    # Sidebar footer
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        ExoPattern v3.0<br>
        5-fold CV / 150 Kepler targets<br>
        scikit-learn + TensorFlow/Keras
    </div>
    """, unsafe_allow_html=True)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Analysis", "Comparison", "Periodogram", "Model Inspector", "Reference"
    ])

    # === Tab 1: Analysis ===
    with tab1:
        section_header("Upload and Analyze Light Curve")

        uploaded_file = st.file_uploader(
            "Choose a light curve file (FITS or CSV)",
            type=['fits', 'fit', 'csv'],
            help="Upload a stellar light curve file in FITS or CSV format"
        )

        if uploaded_file is not None:
            st.markdown(f'<div class="success-banner">File loaded: {uploaded_file.name}</div>',
                        unsafe_allow_html=True)

            if st.button("Analyze Light Curve", type="primary"):
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
                        window_size,
                        model_name=selected_model,
                        feature_groups=feature_groups,
                    )

                if results:
                    st.session_state.analysis_done = True
                    st.session_state.results = results
                    st.session_state.uploaded_file = uploaded_file

            # Display results
            if st.session_state.analysis_done and st.session_state.results:
                results = st.session_state.results
                df = results['df']

                st.markdown('<div class="success-banner">Analysis complete</div>',
                            unsafe_allow_html=True)

                # Summary metric cards
                section_header("Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(metric_card(
                        "Observations", f"{results['stats']['n_points']:,}", accent="blue"
                    ), unsafe_allow_html=True)
                with col2:
                    st.markdown(metric_card(
                        "Anomalies", str(results['n_anomalies']), accent="red"
                    ), unsafe_allow_html=True)
                with col3:
                    st.markdown(metric_card(
                        "Anomaly Rate", f"{results['anomaly_rate']*100:.2f}%", accent="amber"
                    ), unsafe_allow_html=True)
                with col4:
                    st.markdown(metric_card(
                        "Transit Events", str(len(results['transit_events'])), accent="green"
                    ), unsafe_allow_html=True)

                # Evaluation metrics panel
                if results.get('eval_metrics'):
                    m = results['eval_metrics']
                    prec = m['precision']
                    rec = m['recall']
                    f1 = m['f1']
                    roc = m.get('roc_auc', float('nan'))
                    pr = m.get('pr_auc', float('nan'))

                    def fmt(v):
                        return f"{v:.3f}" if not np.isnan(v) else "N/A"

                    metrics_html = '<div class="eval-panel">'
                    metrics_html += '<div class="panel-title">Evaluation Metrics (Ground Truth Available)</div>'
                    metrics_html += '<div class="eval-metric-row">'
                    for label, val in [("Precision", prec), ("Recall", rec), ("F1 Score", f1),
                                       ("ROC-AUC", roc), ("PR-AUC", pr)]:
                        cls = eval_score_class(val)
                        metrics_html += f"""
                        <div class="eval-metric">
                            <div class="label">{label}</div>
                            <div class="value {cls}">{fmt(val)}</div>
                        </div>"""
                    metrics_html += '</div>'

                    # Mini confusion matrix
                    tn, fp, fn, tp = m.get('tn', 0), m.get('fp', 0), m.get('fn', 0), m.get('tp', 0)
                    metrics_html += f"""
                    <div style="margin-top: 1rem;">
                        <div class="panel-title">Confusion Matrix</div>
                        <table class="data-table" style="max-width: 300px;">
                            <tr><th></th><th>Pred Normal</th><th>Pred Anomaly</th></tr>
                            <tr><td style="font-weight:600;color:#8b949e;">True Normal</td>
                                <td>{tn}</td><td>{fp}</td></tr>
                            <tr><td style="font-weight:600;color:#8b949e;">True Anomaly</td>
                                <td>{fn}</td><td>{tp}</td></tr>
                        </table>
                    </div>"""

                    metrics_html += '</div>'
                    st.markdown(metrics_html, unsafe_allow_html=True)

                # Main plot
                section_header("Light Curve Visualization")
                fig = create_lightcurve_plot(df, results['anomaly_mask'])
                st.plotly_chart(fig, use_container_width=True)

                # Detailed dashboard
                section_header("Detailed Analysis Dashboard")
                dashboard_fig = create_analysis_dashboard(df, results)
                st.plotly_chart(dashboard_fig, use_container_width=True)

                # Anomaly details
                col1, col2 = st.columns(2)

                with col1:
                    section_header("Point Anomalies")
                    point_anom = results['point_anomalies']
                    pa_cols = st.columns(3)
                    with pa_cols[0]:
                        st.markdown(metric_card("Total", str(point_anom['n_anomalies']), accent="blue"),
                                    unsafe_allow_html=True)
                    with pa_cols[1]:
                        st.markdown(metric_card("Dips", str(point_anom['n_dips']), accent="red"),
                                    unsafe_allow_html=True)
                    with pa_cols[2]:
                        st.markdown(metric_card("Spikes", str(point_anom['n_spikes']), accent="amber"),
                                    unsafe_allow_html=True)

                with col2:
                    section_header("Transit Events")
                    if results['transit_events']:
                        events_data = []
                        for event in results['transit_events'][:10]:
                            events_data.append({
                                'Start': f"{event['start_time']:.2f}",
                                'End': f"{event['end_time']:.2f}",
                                'Duration': f"{event['time_duration']:.2f}",
                                'Depth (%)': f"{event['depth']*100:.2f}",
                            })
                        st.dataframe(pd.DataFrame(events_data), use_container_width=True, hide_index=True)
                    else:
                        empty_state("No transit events detected")

                # Light curve statistics (folded in from old Statistics tab)
                with st.expander("Light Curve Statistics"):
                    stats = results['stats']
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        time_html = """
                        <table class="data-table">
                            <tr><th colspan="2">Time Information</th></tr>
                            <tr><td>Start Time</td><td>{:.2f}</td></tr>
                            <tr><td>End Time</td><td>{:.2f}</td></tr>
                            <tr><td>Time Span</td><td>{:.2f}</td></tr>
                            <tr><td>Number of Points</td><td>{:,}</td></tr>
                        </table>
                        """.format(stats['time_start'], stats['time_end'],
                                   stats['time_span'], stats['n_points'])
                        st.markdown(time_html, unsafe_allow_html=True)
                    with stat_col2:
                        flux_html = """
                        <table class="data-table">
                            <tr><th colspan="2">Flux Statistics</th></tr>
                            <tr><td>Mean</td><td>{:.4f}</td></tr>
                            <tr><td>Median</td><td>{:.4f}</td></tr>
                            <tr><td>Std Dev</td><td>{:.4f}</td></tr>
                            <tr><td>Min</td><td>{:.4f}</td></tr>
                            <tr><td>Max</td><td>{:.4f}</td></tr>
                        </table>
                        """.format(stats['flux_mean'], stats['flux_median'],
                                   stats['flux_std'], stats['flux_min'], stats['flux_max'])
                        st.markdown(flux_html, unsafe_allow_html=True)

                    # Flux distribution mini-chart
                    fig_flux = go.Figure()
                    fig_flux.add_trace(go.Histogram(
                        x=df['flux'], nbinsx=80,
                        marker_color=COLORS['accent_blue'], opacity=0.8, name='Flux'
                    ))
                    fig_flux.update_layout(xaxis_title='Flux', yaxis_title='Count', showlegend=False)
                    apply_chart_theme(fig_flux, height=250)
                    st.plotly_chart(fig_flux, use_container_width=True)

                # Export
                section_header("Export Results")
                export_df = df.copy()
                export_df['is_anomaly'] = results['anomaly_mask']
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Results (CSV)",
                    data=csv,
                    file_name=f"anomaly_results_{uploaded_file.name}.csv",
                    mime="text/csv"
                )

    # === Tab 2: Model Comparison ===
    with tab2:
        section_header("Multi-Model Comparison")
        st.markdown(
            '<div class="callout-box">Run the same light curve through multiple models to compare '
            'their anomaly detection performance side by side. Requires an analyzed light curve from '
            'the Analysis tab.</div>',
            unsafe_allow_html=True
        )

        if not (st.session_state.analysis_done and st.session_state.results):
            empty_state("Analyze a light curve in the Analysis tab first")
        else:
            results = st.session_state.results
            df = results['df']
            has_labels = results.get('has_labels', False)

            # Model selection for comparison
            all_models = list_models(include_deep=True)
            model_names_display = {name: get_display_name(name) for name in all_models}

            compare_models = st.multiselect(
                "Select models to compare",
                options=all_models,
                default=['isolation_forest', 'lof', 'ensemble'],
                format_func=lambda x: model_names_display[x],
                help="Choose 2 or more models to compare on the same data"
            )

            if len(compare_models) < 2:
                st.warning("Select at least 2 models to compare.")
            elif st.button("Run Comparison", type="primary"):
                if not st.session_state.model_loaded:
                    detector, preprocessor = load_model()
                    st.session_state.detector = detector
                    st.session_state.preprocessor = preprocessor

                preprocessor = st.session_state.preprocessor
                detector = st.session_state.detector

                comparison_results = {}
                progress_bar = st.progress(0)

                for i, model_name in enumerate(compare_models):
                    progress_bar.progress((i) / len(compare_models),
                                         text=f"Running {model_names_display[model_name]}...")

                    # Extract features (reuse from analysis)
                    features = preprocessor.extract_features(df.copy(), window_size, feature_groups)

                    # Get predictions
                    pretrained = load_pretrained(model_name, ARTIFACTS_DIR)
                    if pretrained is not None:
                        preds = pretrained.predict(features)
                        scores = pretrained.score_samples(features) if hasattr(pretrained, 'score_samples') else np.zeros(len(features))
                    else:
                        model = get_model(model_name, contamination=contamination)
                        model.fit(features)
                        preds = model.predict(features)
                        scores = model.score_samples(features) if hasattr(model, 'score_samples') else np.zeros(len(features))

                    stride = max(1, window_size // 4)
                    anomaly_mask, point_scores = map_window_predictions_to_points(
                        preds, scores, len(df), window_size, stride, method='vote'
                    )

                    result_entry = {
                        'anomaly_mask': anomaly_mask,
                        'n_anomalies': int(np.sum(anomaly_mask)),
                        'anomaly_rate': float(np.mean(anomaly_mask)),
                    }

                    # Eval metrics if labels available
                    if has_labels and 'label' in df.columns:
                        evaluator = AnomalyEvaluator()
                        true_labels = df['label'].values
                        y_pred = anomaly_mask.astype(int)[:len(true_labels)]
                        y_true = true_labels[:len(y_pred)]
                        y_scores = -point_scores[:len(y_pred)]
                        result_entry['metrics'] = evaluator.compute_metrics(y_true, y_pred, y_scores)

                    comparison_results[model_name] = result_entry

                progress_bar.progress(1.0, text="Comparison complete")
                st.session_state.comparison_results = comparison_results

            # Display comparison results
            if 'comparison_results' in st.session_state and st.session_state.comparison_results:
                comp = st.session_state.comparison_results

                # Metrics comparison table
                section_header("Metrics Comparison")

                if has_labels and any('metrics' in v for v in comp.values()):
                    table_data = []
                    for model_name, res in comp.items():
                        m = res.get('metrics', {})
                        table_data.append({
                            'Model': model_names_display.get(model_name, model_name),
                            'Precision': f"{m.get('precision', 0):.3f}",
                            'Recall': f"{m.get('recall', 0):.3f}",
                            'F1': f"{m.get('f1', 0):.3f}",
                            'ROC-AUC': f"{m.get('roc_auc', float('nan')):.3f}" if not np.isnan(m.get('roc_auc', float('nan'))) else "N/A",
                            'PR-AUC': f"{m.get('pr_auc', float('nan')):.3f}" if not np.isnan(m.get('pr_auc', float('nan'))) else "N/A",
                            'Anomalies': res['n_anomalies'],
                            'Rate (%)': f"{res['anomaly_rate']*100:.2f}",
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

                    # F1 score bar chart
                    section_header("F1 Score Comparison")
                    f1_fig = go.Figure()
                    model_labels = [model_names_display.get(n, n) for n in comp.keys()]
                    f1_values = [comp[n].get('metrics', {}).get('f1', 0) for n in comp.keys()]

                    colors = [COLORS['accent_green'] if v == max(f1_values) else COLORS['accent_blue']
                              for v in f1_values]

                    f1_fig.add_trace(go.Bar(
                        x=model_labels, y=f1_values,
                        marker_color=colors,
                        text=[f"{v:.3f}" for v in f1_values],
                        textposition='outside',
                        textfont=dict(color=COLORS['text_primary']),
                    ))
                    f1_fig.update_layout(
                        yaxis_title='F1 Score', xaxis_title='',
                        yaxis_range=[0, min(1.0, max(f1_values) * 1.3 + 0.05)],
                        showlegend=False,
                    )
                    apply_chart_theme(f1_fig, height=350)
                    st.plotly_chart(f1_fig, use_container_width=True)

                else:
                    # No labels — just show detection counts
                    table_data = []
                    for model_name, res in comp.items():
                        table_data.append({
                            'Model': model_names_display.get(model_name, model_name),
                            'Anomalies Detected': res['n_anomalies'],
                            'Anomaly Rate (%)': f"{res['anomaly_rate']*100:.2f}",
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

                # Overlaid light curve comparison
                section_header("Anomaly Overlay")
                st.caption("Each model's detected anomalies shown on the same light curve")

                overlay_fig = go.Figure()
                # Base light curve
                overlay_fig.add_trace(go.Scatter(
                    x=df['time'], y=df['flux'],
                    mode='markers', name='Light Curve',
                    marker=dict(color=COLORS['text_secondary'], size=2, opacity=0.3),
                    hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<extra></extra>'
                ))

                # Each model's anomalies in a different color
                overlay_colors = [COLORS['accent_red'], COLORS['accent_blue'],
                                  COLORS['accent_green'], COLORS['accent_amber'],
                                  COLORS['purple'], '#e0e0e0', '#ff79c6', '#8be9fd']
                for idx, (model_name, res) in enumerate(comp.items()):
                    mask = res['anomaly_mask']
                    color = overlay_colors[idx % len(overlay_colors)]
                    overlay_fig.add_trace(go.Scatter(
                        x=df['time'][mask], y=df['flux'][mask],
                        mode='markers',
                        name=model_names_display.get(model_name, model_name),
                        marker=dict(color=color, size=5, symbol='circle', opacity=0.7),
                        hovertemplate=f'{model_names_display.get(model_name, model_name)}<br>'
                                      'Time: %{x:.2f}<br>Flux: %{y:.2f}<extra></extra>'
                    ))

                overlay_fig.update_layout(
                    xaxis_title='Time', yaxis_title='Flux', showlegend=True,
                )
                apply_chart_theme(overlay_fig, height=500)
                st.plotly_chart(overlay_fig, use_container_width=True)

                # Agreement analysis
                section_header("Model Agreement")
                st.caption("How many models flag each point as anomalous")

                all_masks = np.array([comp[n]['anomaly_mask'] for n in comp.keys()])
                agreement_count = all_masks.sum(axis=0)
                n_models = len(comp)

                agreement_fig = go.Figure()
                agreement_fig.add_trace(go.Scatter(
                    x=df['time'], y=df['flux'],
                    mode='markers',
                    marker=dict(
                        color=agreement_count,
                        colorscale=[[0, COLORS['text_secondary']],
                                    [0.5, COLORS['accent_amber']],
                                    [1, COLORS['accent_red']]],
                        cmin=0, cmax=n_models,
                        size=np.where(agreement_count > 0, 5, 2),
                        opacity=np.where(agreement_count > 0, 0.9, 0.2),
                        colorbar=dict(
                            title=dict(text="Models<br>Agreeing", font=dict(size=11)),
                            tickfont=dict(size=10),
                            thickness=12,
                        ),
                    ),
                    hovertemplate='Time: %{x:.2f}<br>Flux: %{y:.2f}<br>'
                                  'Models flagging: %{marker.color}/' + str(n_models) + '<extra></extra>'
                ))
                agreement_fig.update_layout(
                    xaxis_title='Time', yaxis_title='Flux', showlegend=False,
                )
                apply_chart_theme(agreement_fig, height=400)
                st.plotly_chart(agreement_fig, use_container_width=True)

                # Consensus metrics
                unanimous = int(np.sum(agreement_count == n_models))
                majority = int(np.sum(agreement_count > n_models / 2))
                any_flag = int(np.sum(agreement_count > 0))

                cons_cols = st.columns(3)
                with cons_cols[0]:
                    st.markdown(metric_card(
                        "Unanimous", str(unanimous),
                        f"All {n_models} models agree", accent="red"
                    ), unsafe_allow_html=True)
                with cons_cols[1]:
                    st.markdown(metric_card(
                        "Majority", str(majority),
                        f">{n_models//2} models agree", accent="amber"
                    ), unsafe_allow_html=True)
                with cons_cols[2]:
                    st.markdown(metric_card(
                        "Any Model", str(any_flag),
                        "At least 1 model flags", accent="blue"
                    ), unsafe_allow_html=True)

    # === Tab 3: Periodogram ===
    with tab3:
        if st.session_state.analysis_done and st.session_state.results:
            results = st.session_state.results
            df = results['df']

            preprocessor = st.session_state.preprocessor
            time_arr = df['time'].values
            flux_arr = df['flux'].values

            try:
                frequencies, power = preprocessor.compute_periodogram(time_arr, flux_arr)
                periods = 1.0 / (frequencies + 1e-12)

                section_header("Lomb-Scargle Periodogram")
                fig_pg = go.Figure()
                fig_pg.add_trace(go.Scatter(
                    x=periods, y=power, mode='lines',
                    name='Lomb-Scargle Power',
                    line=dict(color=COLORS['accent_blue'], width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(88, 166, 255, 0.08)',
                ))
                fig_pg.update_layout(
                    xaxis_title='Period (days)',
                    yaxis_title='Power',
                    xaxis_type='log',
                    showlegend=False,
                )
                apply_chart_theme(fig_pg, height=400)
                st.plotly_chart(fig_pg, use_container_width=True)

                # Dominant periods as table
                section_header("Top Detected Periods")
                top_idx = np.argsort(power)[::-1][:5]
                periods_data = []
                for rank, idx in enumerate(top_idx):
                    periods_data.append({
                        'Rank': f"#{rank+1}",
                        'Period (days)': f"{periods[idx]:.3f}",
                        'Power': f"{power[idx]:.4f}",
                    })
                st.dataframe(pd.DataFrame(periods_data), use_container_width=True, hide_index=True)

                # Periodogram features as table
                section_header("Periodogram Features")
                pg_features = preprocessor.extract_global_periodogram_features(time_arr, flux_arr)
                feat_names = ['LS Max Power', 'LS Dominant Period', 'LS Second Power',
                              'LS Power Ratio', 'BLS Power', 'BLS Period', 'BLS Duration']
                feat_data = [{'Feature': name, 'Value': f"{val:.4f}"}
                             for name, val in zip(feat_names, pg_features)]
                st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Periodogram computation failed: {e}")
        else:
            empty_state("Upload and analyze a light curve first to see periodogram analysis")

    # === Tab 4: Model Inspector ===
    with tab4:
        section_header("Available Models")

        # Check which models have pre-trained artifacts
        import os
        import json

        artifacts_path = Path(ARTIFACTS_DIR)
        training_meta = {}
        meta_file = artifacts_path / 'training_metadata.json'
        if meta_file.exists():
            with open(meta_file) as f:
                training_meta = json.load(f)

        # Build model inventory
        all_models = list_models(include_deep=True)
        inventory_data = []
        for name in all_models:
            display = get_display_name(name)
            mtype = "Deep Learning" if name in DEEP_MODELS else "Classical"

            # Check artifact availability
            has_artifact = False
            artifact_size = ""
            if name in CLASSICAL_MODELS and name != 'bls':
                pkl = artifacts_path / f"{name}.pkl"
                if pkl.exists():
                    has_artifact = True
                    size_mb = pkl.stat().st_size / (1024 * 1024)
                    artifact_size = f"{size_mb:.1f} MB"
            elif name in DEEP_MODELS:
                h5 = artifacts_path / f"{name}.h5"
                if h5.exists():
                    has_artifact = True
                    size_mb = h5.stat().st_size / (1024 * 1024)
                    artifact_size = f"{size_mb:.1f} MB"

            inventory_data.append({
                'Model': display,
                'Type': mtype,
                'Pre-trained': 'Yes' if has_artifact else 'No',
                'Artifact Size': artifact_size if has_artifact else '--',
            })

        st.dataframe(pd.DataFrame(inventory_data), use_container_width=True, hide_index=True)

        # Training metadata
        if training_meta:
            section_header("Training Data Summary")
            meta_cols = st.columns(4)
            with meta_cols[0]:
                st.markdown(metric_card(
                    "Kepler Targets", str(training_meta.get('n_targets', '?')), accent="blue"
                ), unsafe_allow_html=True)
            with meta_cols[1]:
                st.markdown(metric_card(
                    "Windows", f"{training_meta.get('n_windows', 0):,}", accent="green"
                ), unsafe_allow_html=True)
            with meta_cols[2]:
                st.markdown(metric_card(
                    "Features", str(training_meta.get('n_features', '?')), accent="purple"
                ), unsafe_allow_html=True)
            with meta_cols[3]:
                st.markdown(metric_card(
                    "Anomaly Fraction",
                    f"{training_meta.get('anomaly_fraction', 0)*100:.1f}%",
                    accent="amber"
                ), unsafe_allow_html=True)

            # Training config details
            with st.expander("Training Configuration"):
                config_data = {
                    'Window Size': training_meta.get('window_size', '?'),
                    'Contamination': training_meta.get('contamination', '?'),
                    'Random Seed': training_meta.get('random_seed', '?'),
                    'Feature Groups': ', '.join(training_meta.get('feature_groups', [])),
                    'Models Trained': ', '.join(training_meta.get('models_trained', [])),
                }
                config_df = pd.DataFrame([{'Parameter': k, 'Value': str(v)}
                                          for k, v in config_data.items()])
                st.dataframe(config_df, use_container_width=True, hide_index=True)

        # Currently selected model details
        section_header(f"Selected Model: {model_display[selected_model]}")

        # Model-specific details
        model_details = {
            'isolation_forest': {
                'Algorithm': 'Isolation Forest',
                'Strategy': 'Isolates anomalies by randomly partitioning feature space via trees',
                'Key Params': 'n_estimators=100, contamination=0.1',
                'Strengths': 'Fast, scales well, handles high-dimensional data',
                'Limitations': 'Assumes anomalies are few and different',
            },
            'lof': {
                'Algorithm': 'Local Outlier Factor',
                'Strategy': 'Measures local density deviation compared to k-nearest neighbors',
                'Key Params': 'n_neighbors=20, contamination=0.1, novelty=True',
                'Strengths': 'Detects local outliers in varying density regions',
                'Limitations': 'Sensitive to n_neighbors choice, slower on large datasets',
            },
            'ocsvm': {
                'Algorithm': 'One-Class SVM',
                'Strategy': 'Learns a decision boundary around normal data in kernel space',
                'Key Params': 'kernel=rbf, nu=contamination, gamma=scale',
                'Strengths': 'Effective in high-dimensional spaces, robust boundary',
                'Limitations': 'Computationally expensive for large datasets, sensitive to kernel choice',
            },
            'dbscan': {
                'Algorithm': 'DBSCAN',
                'Strategy': 'Density-based clustering; points not in any cluster are anomalies',
                'Key Params': 'eps=0.5, min_samples=5',
                'Strengths': 'No assumption on cluster shape, naturally identifies outliers',
                'Limitations': 'Sensitive to eps/min_samples, struggles with varying densities',
            },
            'ensemble': {
                'Algorithm': 'Ensemble (IF+LOF)',
                'Strategy': 'AND-logic: both Isolation Forest and LOF must agree a point is anomalous',
                'Key Params': 'Inherits IF and LOF params, AND combination',
                'Strengths': 'Reduces false positives by requiring model agreement',
                'Limitations': 'May miss anomalies that only one model catches',
            },
            'bls': {
                'Algorithm': 'Box Least Squares',
                'Strategy': 'Searches for periodic box-shaped transit signals in time-domain',
                'Key Params': 'Period search range, duration grid',
                'Strengths': 'Domain-specific for transit detection, physically motivated',
                'Limitations': 'Only detects periodic transits, no pre-trained artifact',
            },
            'cnn_autoencoder': {
                'Algorithm': 'CNN Autoencoder',
                'Strategy': 'Learns to reconstruct normal patterns; high reconstruction error = anomaly',
                'Architecture': 'Conv1D(32) -> Conv1D(16) -> Conv1D(8) -> Dense(8) -> Decoder',
                'Key Params': 'latent_dim=8, threshold from training',
                'Strengths': 'Captures spatial patterns in feature windows',
                'Limitations': 'Requires GPU for training, threshold-sensitive',
            },
            'lstm_autoencoder': {
                'Algorithm': 'LSTM Autoencoder',
                'Strategy': 'Sequence-based reconstruction; captures temporal dependencies',
                'Architecture': 'LSTM(64) -> LSTM(32) -> Dense(32) -> RepeatVector -> LSTM(32) -> LSTM(64)',
                'Key Params': 'latent_dim=32, seq_length=10',
                'Strengths': 'Models temporal patterns across window sequences',
                'Limitations': 'Requires GPU, longer training time, sequence padding needed',
            },
        }

        details = model_details.get(selected_model, {})
        if details:
            detail_html = '<table class="data-table">'
            for key, val in details.items():
                detail_html += f'<tr><td style="font-weight:600;color:#8b949e;width:140px;">{key}</td><td>{val}</td></tr>'
            detail_html += '</table>'
            st.markdown(detail_html, unsafe_allow_html=True)

        # Deep model metadata
        if selected_model in DEEP_MODELS:
            deep_meta_file = artifacts_path / f"{selected_model}_meta.json"
            if deep_meta_file.exists():
                with open(deep_meta_file) as f:
                    deep_meta = json.load(f)
                section_header("Model Metadata")
                deep_meta_df = pd.DataFrame([{'Parameter': k, 'Value': str(v)}
                                              for k, v in deep_meta.items()])
                st.dataframe(deep_meta_df, use_container_width=True, hide_index=True)

        # Feature groups detail
        section_header("Feature Groups (38 total)")
        for group_name, feature_list in FEATURE_GROUPS.items():
            with st.expander(f"{group_name.capitalize()} ({len(feature_list)} features)"):
                feat_df = pd.DataFrame([{'Feature': f} for f in feature_list])
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # === Tab 5: Reference ===
    with tab5:
        # Abstract-style overview
        st.markdown("""
        <div class="callout-box">
            <strong>ExoPattern</strong> is a conference-paper-quality ML system for detecting exoplanet transits
            in Kepler/TESS light curves. It compares <strong>8 anomaly detection models</strong> with
            <strong>38 engineered features</strong> across 5 feature groups, evaluated via 5-fold
            cross-validation with ground truth labels from NASA Exoplanet Archive ephemerides.
        </div>
        """, unsafe_allow_html=True)

        # Detection targets
        section_header("Detection Targets")
        targets_html = """
        <table class="data-table">
            <tr><th>Target</th><th>Description</th></tr>
            <tr><td>Exoplanet Transits</td><td>Periodic dips in brightness caused by planets passing in front of stars</td></tr>
            <tr><td>Stellar Flares</td><td>Sudden brightness increases from stellar activity</td></tr>
            <tr><td>Noise Artifacts</td><td>Instrumental or environmental anomalies</td></tr>
            <tr><td>Other Patterns</td><td>Any deviation from normal stellar behavior</td></tr>
        </table>
        """
        st.markdown(targets_html, unsafe_allow_html=True)

        # Design decisions
        section_header("Key Design Decisions")
        decisions_html = """
        <table class="data-table">
            <tr><th>Decision</th><th>Details</th></tr>
            <tr><td>Window-to-point mapping</td><td>Majority voting (&gt;50% of overlapping windows must flag a point)</td></tr>
            <tr><td>Ensemble logic</td><td>AND (both IF and LOF must agree) to reduce false positives</td></tr>
            <tr><td>Pre-trained models</td><td>Classical models trained on 150 Kepler planet hosts (181k windows)</td></tr>
            <tr><td>Ground truth</td><td>NASA Exoplanet Archive ephemerides for transit labels</td></tr>
            <tr><td>Window labels</td><td>max &gt; 0 (any transit point in window = anomalous)</td></tr>
        </table>
        """
        st.markdown(decisions_html, unsafe_allow_html=True)

        # Usage guide
        section_header("Quick Start")
        guide_html = """
        <table class="data-table">
            <tr><th>Step</th><th>Action</th></tr>
            <tr><td>1</td><td>Upload a Kepler/TESS light curve file (FITS or CSV) in the Analysis tab</td></tr>
            <tr><td>2</td><td>Select a detection model from the sidebar (Ensemble recommended)</td></tr>
            <tr><td>3</td><td>Click "Analyze Light Curve" — pre-trained models give instant results</td></tr>
            <tr><td>4</td><td>Compare models in the Comparison tab to see agreement across methods</td></tr>
            <tr><td>5</td><td>Upload files with a <code>label</code> column to see evaluation metrics</td></tr>
        </table>
        """
        st.markdown(guide_html, unsafe_allow_html=True)

        # Version/citation footer
        st.markdown("""
        <div class="sidebar-footer" style="margin-top: 2rem; text-align: center;">
            ExoPattern v3.0 &middot; Built with Streamlit, scikit-learn, TensorFlow/Keras, astropy, plotly
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
