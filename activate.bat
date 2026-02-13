@echo off
REM Jarvis Voice Bot - Activate Virtual Environment

echo Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure you have run setup.bat first
    pause
    exit /b 1
)

echo Virtual environment activated!
echo.
echo You can now run:
echo   python run.py
echo.
