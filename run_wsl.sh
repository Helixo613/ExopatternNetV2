#!/bin/bash
# WSL-specific shell script to run the Stellar Anomaly Detector
# This script is optimized for WSL2 networking

set -e  # Exit on error

echo "========================================"
echo "Stellar Light Curve Anomaly Detector"
echo "         WSL2 Edition"
echo "========================================"
echo ""

# Get WSL IP address
WSL_IP=$(hostname -I | awk '{print $1}')
echo "WSL IP Address: $WSL_IP"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Using Python $PYTHON_VERSION"
echo ""

# Option to skip virtual environment
USE_VENV=true
if [ "$1" = "--no-venv" ]; then
    echo "Running without virtual environment (using system Python)"
    echo ""
    USE_VENV=false
fi

if [ "$USE_VENV" = true ]; then
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."

        # Try to create venv
        if ! python3 -m venv venv 2>&1; then
            echo ""
            echo "ERROR: Failed to create virtual environment"
            echo ""
            echo "On Debian/Ubuntu systems, you need to install python3-venv:"
            echo "  sudo apt update"
            echo "  sudo apt install python3-venv"
            echo ""
            echo "On other systems:"
            echo "  - Fedora/RHEL: sudo dnf install python3-virtualenv"
            echo "  - macOS: python3-venv should be included with Python"
            echo ""
            echo "Alternatively, run with --no-venv to use system Python:"
            echo "  ./run_wsl.sh --no-venv"
            echo ""
            exit 1
        fi

        echo "Virtual environment created successfully"
        echo ""
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    if [ ! -f "venv/bin/activate" ]; then
        echo "ERROR: venv/bin/activate not found"
        echo "The virtual environment may be corrupted. Try:"
        echo "  rm -rf venv"
        echo "  ./run_wsl.sh"
        exit 1
    fi

    source venv/bin/activate
    echo ""

    # Set Python command to use venv's python
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    # Use system Python
    PYTHON_CMD="python3"
    PIP_CMD="pip3"

    # Check if pip3 is available
    if ! command -v pip3 &> /dev/null; then
        echo "ERROR: pip3 is not installed"
        echo "Please install pip for Python 3:"
        echo "  sudo apt install python3-pip"
        exit 1
    fi
fi

# Check if requirements are installed
echo "Checking dependencies..."
if ! $PIP_CMD show streamlit > /dev/null 2>&1; then
    echo "Installing dependencies..."
    echo "This may take several minutes..."

    if ! $PIP_CMD install -r requirements.txt; then
        echo ""
        echo "ERROR: Failed to install dependencies"
        echo "Please check your internet connection and try again"
        exit 1
    fi
    echo ""
    echo "Dependencies installed successfully"
    echo ""
fi

# Check if sample data exists
if [ ! -f "data/samples/normal_star.csv" ]; then
    echo "Generating sample data..."

    if [ ! -f "generate_sample_data.py" ]; then
        echo "WARNING: generate_sample_data.py not found, skipping sample data generation"
        echo ""
    else
        if ! $PYTHON_CMD generate_sample_data.py; then
            echo ""
            echo "WARNING: Failed to generate sample data"
            echo "The application may still work with your own data"
            echo ""
        else
            echo "Sample data generated successfully"
            echo ""
        fi
    fi
fi

# Launch the application with WSL-optimized settings
echo "Launching Streamlit application (WSL mode)..."
echo ""
echo "========================================"
echo "IMPORTANT: Access URLs"
echo "========================================"
echo ""
echo "Try these URLs in your Windows browser (in order):"
echo ""
echo "  1. http://localhost:8501"
echo "  2. http://${WSL_IP}:8501"
echo ""
echo "If localhost doesn't work, use the WSL IP address above."
echo ""
echo "Press Ctrl+C to stop the application"
echo "========================================"
echo ""

if ! command -v streamlit &> /dev/null; then
    echo "ERROR: streamlit command not found"
    echo "Dependencies may not be installed correctly"
    exit 1
fi

# Run Streamlit with explicit server settings for WSL
streamlit run frontend/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
