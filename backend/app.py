"""
Main entry point for the Flask API server.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api import create_app

if __name__ == '__main__':
    app = create_app()
    print("\n" + "="*50)
    print("Stellar Light Curve Anomaly Detection API")
    print("="*50)
    print("\nAPI Server starting on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  - GET  /health              - Health check")
    print("  - POST /api/analyze         - Analyze light curve")
    print("  - POST /api/train           - Train new model")
    print("  - POST /api/train/synthetic - Train on synthetic data")
    print("  - POST /api/export          - Export results to CSV")
    print("\n" + "="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
