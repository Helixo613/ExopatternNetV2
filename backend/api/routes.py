"""
Flask API routes for anomaly detection service.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging
import traceback
from typing import Dict, Any

from backend.data import LightCurveLoader
from backend.ml import LightCurvePreprocessor, AnomalyDetector, ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config: Dict[str, Any] = None) -> Flask:
    """
    Create and configure Flask application.
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for frontend communication

    # Configuration
    if config:
        app.config.update(config)

    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max file size
    app.config['UPLOAD_FOLDER'] = Path('backend/uploads')
    app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)

    # Initialize components
    loader = LightCurveLoader()
    preprocessor = LightCurvePreprocessor()
    trainer = ModelTrainer()

    # Try to load pre-trained model
    detector = None
    try:
        detector = trainer.load_model('default_model')
        logger.info("Loaded pre-trained model")
    except:
        logger.warning("No pre-trained model found. Train a model or use synthetic training.")

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'model_loaded': detector is not None and detector.is_fitted if detector else False
        })

    @app.route('/api/analyze', methods=['POST'])
    def analyze_lightcurve():
        """
        Analyze uploaded light curve for anomalies.

        Expects:
            - file: Light curve file (FITS or CSV)
            - contamination: (optional) Expected anomaly rate
            - method: (optional) Detection method
        """
        try:
            # Check if file is present
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']

            if file.filename == '':
                return jsonify({'error': 'Empty filename'}), 400

            # Get optional parameters
            contamination = float(request.form.get('contamination', 0.1))
            method = request.form.get('method', 'ensemble')
            window_size = int(request.form.get('window_size', 50))

            # Save uploaded file temporarily
            temp_path = app.config['UPLOAD_FOLDER'] / file.filename
            file.save(str(temp_path))

            logger.info(f"Processing file: {file.filename}")

            # Load data
            df = loader.load_file(str(temp_path))
            logger.info(f"Loaded {len(df)} data points")

            # Get summary statistics
            stats = loader.get_summary_stats(df)

            # Preprocess
            df_processed = preprocessor.preprocess(df, normalize=True)

            # Extract features
            features = preprocessor.extract_features(df_processed, window_size)

            # Detect anomalies
            nonlocal detector

            # If no model is loaded, train on this data
            if detector is None or not detector.is_fitted:
                logger.info("Training new model on uploaded data...")
                detector = trainer.train_from_dataframe(df, contamination, window_size)

            # Predict anomalies
            predictions, scores = detector.predict_with_scores(features)

            # Map window predictions back to original points
            # Simple approach: mark points in anomalous windows
            stride = max(1, window_size // 4)
            anomaly_mask = np.zeros(len(df_processed), dtype=bool)

            for i, pred in enumerate(predictions):
                if pred == -1:  # Anomaly
                    start_idx = i * stride
                    end_idx = min(start_idx + window_size, len(df_processed))
                    anomaly_mask[start_idx:end_idx] = True

            anomaly_indices = np.where(anomaly_mask)[0]

            # Additional detection: point anomalies
            point_anomalies = detector.detect_point_anomalies(df_processed, threshold=3.0)

            # Additional detection: transit events
            transit_events = detector.detect_transit_events(df_processed, depth_threshold=0.01, duration_min=3)

            # Prepare response
            response = {
                'success': True,
                'filename': file.filename,
                'n_points': len(df),
                'statistics': stats,
                'anomalies': {
                    'window_based': {
                        'n_windows': len(predictions),
                        'n_anomalous_windows': int(np.sum(predictions == -1)),
                        'anomaly_rate': float(np.sum(predictions == -1) / len(predictions)),
                        'anomaly_indices': anomaly_indices.tolist(),
                        'anomaly_scores': scores.tolist(),
                    },
                    'point_anomalies': point_anomalies,
                    'transit_events': transit_events,
                },
                'data': {
                    'time': df_processed['time'].tolist(),
                    'flux': df_processed['flux'].tolist(),
                    'flux_err': df_processed['flux_err'].tolist(),
                    'anomaly_mask': anomaly_mask.tolist(),
                }
            }

            # Clean up
            temp_path.unlink()

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error analyzing light curve: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }), 500

    @app.route('/api/train', methods=['POST'])
    def train_model():
        """
        Train a new anomaly detection model.

        Expects:
            - files: Multiple light curve files for training
            - contamination: Expected anomaly rate
        """
        try:
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400

            files = request.files.getlist('files')
            contamination = float(request.form.get('contamination', 0.1))
            window_size = int(request.form.get('window_size', 50))

            if not files:
                return jsonify({'error': 'No files selected'}), 400

            # Save files temporarily
            temp_paths = []
            for file in files:
                if file.filename:
                    temp_path = app.config['UPLOAD_FOLDER'] / file.filename
                    file.save(str(temp_path))
                    temp_paths.append(str(temp_path))

            logger.info(f"Training on {len(temp_paths)} files...")

            # Train model
            nonlocal detector
            detector = trainer.train_from_files(temp_paths, contamination, window_size)

            # Clean up
            for temp_path in temp_paths:
                Path(temp_path).unlink()

            return jsonify({
                'success': True,
                'message': f'Model trained on {len(temp_paths)} light curves',
                'model_info': {
                    'contamination': contamination,
                    'window_size': window_size,
                }
            })

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/train/synthetic', methods=['POST'])
    def train_synthetic():
        """
        Train model on synthetic data.

        Useful for initializing a baseline model.
        """
        try:
            n_samples = int(request.json.get('n_samples', 100))
            contamination = float(request.json.get('contamination', 0.1))

            logger.info(f"Training on {n_samples} synthetic light curves...")

            nonlocal detector
            detector = trainer.train_with_synthetic_data(n_samples, contamination)

            return jsonify({
                'success': True,
                'message': f'Model trained on {n_samples} synthetic light curves',
                'model_info': {
                    'contamination': contamination,
                    'n_samples': n_samples,
                }
            })

        except Exception as e:
            logger.error(f"Error training on synthetic data: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/export', methods=['POST'])
    def export_results():
        """
        Export analysis results to CSV.

        Expects JSON with analysis results.
        """
        try:
            data = request.json

            if 'data' not in data:
                return jsonify({'error': 'No data provided'}), 400

            # Create DataFrame
            df = pd.DataFrame({
                'time': data['data']['time'],
                'flux': data['data']['flux'],
                'flux_err': data['data']['flux_err'],
                'is_anomaly': data['data']['anomaly_mask'],
            })

            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            df.to_csv(temp_file.name, index=False)

            return send_file(
                temp_file.name,
                mimetype='text/csv',
                as_attachment=True,
                download_name='anomaly_results.csv'
            )

        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
