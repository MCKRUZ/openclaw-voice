@echo off
echo ======================================================================
echo Fixing PyTorch CUDA Installation
echo ======================================================================
echo.
echo Current Status:
call venv\Scripts\activate.bat
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
echo.

echo ======================================================================
echo This will:
echo   1. Uninstall CPU-only PyTorch
echo   2. Install CUDA 12.1-enabled PyTorch
echo   3. Verify RTX 5090 is accessible
echo ======================================================================
echo.

set /p continue="Continue? (y/n): "
if /i not "%continue%"=="y" (
    echo Cancelled.
    exit /b 1
)

echo.
echo [1/3] Uninstalling CPU PyTorch...
pip uninstall torch torchaudio torchvision -y

echo.
echo [2/3] Installing CUDA PyTorch (this may take a few minutes)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Verifying installation...
python -c "import torch; print(f'\nPyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ======================================================================
echo Done! Your TTS and STT should now use GPU acceleration.
echo ======================================================================
echo.
echo Next: Run the bot and check performance improvement!
pause
