@echo off
chcp 65001 >nul
setlocal

REM ============================================
REM DR-Saintvision Quick Run Script
REM ============================================

echo.
echo DR-Saintvision - Starting...
echo.

REM Check for virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    REM Try conda
    call conda activate dr-saintvision 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] No environment found. Running with system Python.
    ) else (
        echo Conda environment activated.
    )
)

REM Check arguments
if "%1"=="" goto :full
if "%1"=="api" goto :api
if "%1"=="gradio" goto :gradio
if "%1"=="cli" goto :cli
if "%1"=="help" goto :help
goto :query

:full
echo Starting full system (API + Gradio)...
python main.py
goto :end

:api
echo Starting API server only...
python main.py --api-only
goto :end

:gradio
echo Starting Gradio UI only...
python main.py --gradio-only
goto :end

:cli
echo Starting CLI mode...
python main.py --cli
goto :end

:query
echo Processing query: %*
python main.py --query "%*"
goto :end

:help
echo.
echo Usage: run.bat [option]
echo.
echo Options:
echo   (none)    Start full system (API + Gradio)
echo   api       Start API server only
echo   gradio    Start Gradio UI only
echo   cli       Interactive CLI mode
echo   help      Show this help
echo   "query"   Process a single query
echo.
echo Examples:
echo   run.bat
echo   run.bat api
echo   run.bat cli
echo   run.bat "What is artificial intelligence?"
echo.
goto :end

:end
endlocal
