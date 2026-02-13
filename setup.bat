@echo off
REM Jarvis Voice Bot - Windows Setup Script

echo ======================================================================
echo Jarvis Voice Bot - Setup
echo ======================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.12 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version

REM Create virtual environment
echo.
echo [2/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
)

REM Activate virtual environment
echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo [5/5] Installing dependencies...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
echo.
if exist .env (
    echo .env file already exists, skipping...
) else (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Edit .env file and add your credentials:
    echo   - DISCORD_BOT_TOKEN
    echo   - OPENCLAW_BASE_URL
    echo   - OPENCLAW_AUTH_TOKEN
    echo.
)

REM Create voices directory if it doesn't exist
if not exist server\voices (
    echo Creating voices directory...
    mkdir server\voices
)

REM Create models directory if it doesn't exist
if not exist models (
    echo Creating models directory...
    mkdir models
)

echo.
echo ======================================================================
echo Setup Complete!
echo ======================================================================
echo.
echo Next steps:
echo   1. Edit .env file with your credentials
echo   2. Add voice reference files to server\voices\:
echo      - jarvis.wav (10-30 seconds of clean speech)
echo      - sage.wav (10-30 seconds of clean speech)
echo   3. Run: activate.bat
echo   4. Run: python run.py
echo.
echo For more information, see README.md
echo.
pause
