@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM ============================================
REM DR-Saintvision Windows Setup Script
REM Multi-Agent AI Debate System
REM ============================================

echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║          DR-Saintvision Environment Setup                     ║
echo ║          Multi-Agent AI Debate System                         ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.

REM Check Python version
echo [1/7] Checking Python version...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Check if conda is available
echo.
echo [2/7] Checking Conda...
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Conda not found. Using pip only.
    echo For better package management, install Anaconda/Miniconda.
    set USE_CONDA=0
) else (
    echo Conda found.
    set USE_CONDA=1
)

REM Create virtual environment
echo.
echo [3/7] Setting up environment...
if %USE_CONDA%==1 (
    echo Creating conda environment 'dr-saintvision'...
    conda create -n dr-saintvision python=3.10 -y
    call conda activate dr-saintvision
) else (
    echo Creating virtual environment...
    if not exist "venv" (
        python -m venv venv
    )
    call venv\Scripts\activate.bat
)

REM Check for CUDA/GPU
echo.
echo [4/7] Checking GPU availability...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected!
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader') do echo GPU: %%i
    set HAS_GPU=1
) else (
    echo [WARNING] No NVIDIA GPU detected. Will use CPU (slower).
    set HAS_GPU=0
)

REM Install PyTorch
echo.
echo [5/7] Installing PyTorch...
if %HAS_GPU%==1 (
    echo Installing PyTorch with CUDA 11.8 support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio
)

REM Install requirements
echo.
echo [6/7] Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

REM Create directories and copy config
echo.
echo [7/7] Setting up project structure...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "cache" mkdir cache

if not exist ".env" (
    copy .env.example .env >nul 2>&1
    echo Created .env from template
)

REM Success message
echo.
echo ╔═══════════════════════════════════════════════════════════════╗
echo ║                    Setup Completed!                           ║
echo ╚═══════════════════════════════════════════════════════════════╝
echo.
echo Next steps:
echo.
if %USE_CONDA%==1 (
    echo   1. Activate environment:
    echo      conda activate dr-saintvision
) else (
    echo   1. Activate environment:
    echo      venv\Scripts\activate
)
echo.
echo   2. Download models (choose one):
echo      a) Using Ollama (recommended):
echo         - Install Ollama from https://ollama.ai
echo         - Run: ollama pull mistral:7b-instruct-v0.2-q4_0
echo         - Run: ollama pull llama3.2:latest
echo         - Run: ollama pull qwen2.5:7b-instruct-q4_0
echo.
echo      b) Using Hugging Face:
echo         - Run: python download_models.py
echo.
echo   3. Start the system:
echo      python main.py
echo.
echo   4. Access the interface:
echo      - Gradio UI: http://localhost:7860
echo      - API Docs:  http://localhost:8000/docs
echo.

pause
endlocal
