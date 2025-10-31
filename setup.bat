@echo off
echo ========================================
echo Fraud Analytics Dashboard Setup
echo ========================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment created
echo.

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

echo [3/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

echo [4/4] Generating sample data...
python generate_sample_data.py
if errorlevel 1 (
    echo ERROR: Failed to generate sample data
    pause
    exit /b 1
)
echo ✓ Sample data generated
echo.

echo ========================================
echo Setup Complete! 🎉
echo ========================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env
echo 2. Add your API key to .env
echo 3. Run: streamlit run app.py
echo.
echo Or run start.bat to launch the dashboard
echo.
pause
