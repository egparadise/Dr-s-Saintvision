#!/bin/bash
# ============================================
# DR-Saintvision Quick Run Script
# ============================================

echo ""
echo "DR-Saintvision - Starting..."
echo ""

# Activate environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated."
elif command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate dr-saintvision 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Conda environment activated."
    else
        echo "[WARNING] No environment found. Running with system Python."
    fi
fi

# Handle arguments
case "$1" in
    "")
        echo "Starting full system (API + Gradio)..."
        python main.py
        ;;
    "api")
        echo "Starting API server only..."
        python main.py --api-only
        ;;
    "gradio")
        echo "Starting Gradio UI only..."
        python main.py --gradio-only
        ;;
    "cli")
        echo "Starting CLI mode..."
        python main.py --cli
        ;;
    "help"|"-h"|"--help")
        echo ""
        echo "Usage: ./run.sh [option]"
        echo ""
        echo "Options:"
        echo "  (none)    Start full system (API + Gradio)"
        echo "  api       Start API server only"
        echo "  gradio    Start Gradio UI only"
        echo "  cli       Interactive CLI mode"
        echo "  help      Show this help"
        echo "  \"query\"   Process a single query"
        echo ""
        echo "Examples:"
        echo "  ./run.sh"
        echo "  ./run.sh api"
        echo "  ./run.sh cli"
        echo "  ./run.sh \"What is artificial intelligence?\""
        echo ""
        ;;
    *)
        echo "Processing query: $*"
        python main.py --query "$*"
        ;;
esac
