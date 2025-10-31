@echo off
echo ========================================
echo Starting Fraud Analytics Dashboard...
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo AI Copilot features will be disabled.
    echo Copy .env.example to .env and add your API key to enable AI features.
    echo.
    timeout /t 3
)

REM Check if sample data exists
if not exist "data\sample_transactions.csv" (
    echo Generating sample data...
    python generate_sample_data.py
    echo.
)

REM Start Streamlit
echo Launching dashboard...
echo.
streamlit run app.py

pause
