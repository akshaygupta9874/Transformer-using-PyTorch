@echo off
REM Recreate virtual environment for English-Hindi Transformer NMT
echo [1/3] Detecting Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3.11 is required but was not found via py -3.11.
    exit /b 1
)

echo [2/3] Creating virtual environment in .venv...
py -3.11 -m venv --system-site-packages .venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment.
    exit /b 1
)

echo [3/3] Validating environment and PyTorch CUDA support...
.\.venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
if errorlevel 1 (
    echo Warning: PyTorch check failed. Installing dependencies from requirements.txt...
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo.
echo Environment recreation complete!
echo Activate via: .\.venv\Scripts\activate
echo Or run directly: .\.venv\Scripts\python.exe train.py
