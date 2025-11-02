@echo off
REM Windows batch script to run the Stellar Anomaly Detector

echo ========================================
echo Stellar Light Curve Anomaly Detector
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Check if requirements are installed
echo Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Check if sample data exists
if not exist "data\samples\normal_star.csv" (
    echo Generating sample data...
    python generate_sample_data.py
    echo.
)

REM Launch the application
echo Launching Streamlit application...
echo.
echo The application will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
streamlit run frontend\app.py

pause
