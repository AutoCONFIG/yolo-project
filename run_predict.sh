#!/bin/bash
#
# YOLO Prediction Shell Script
# ==============================
# A convenient wrapper for running YOLO prediction
# with various configurations.
#
# Usage:
#   ./run_predict.sh                                     # Predict with default config
#   ./run_predict.sh --source images/test/               # Override input source
#       ./run_predict.sh --config configs/predict/predict.yaml --source images/test/
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if --input is provided (required for predict mode)
HAS_INPUT=false
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--input" ]] || [[ "$arg" == --input=* ]]; then
        HAS_INPUT=true
        break
    fi
done

if [ "$HAS_INPUT" = false ]; then
    echo "Error: --input is required for prediction mode"
    echo "Usage: ./run_predict.sh --input <path_or_url> [--config <yaml>]"
    echo ""
    echo "Examples:"
    echo "  ./run_predict.sh --input images/test/"
    echo "  ./run_predict.sh --input video.mp4 --config configs/predict/predict.yaml"
    echo "  ./run_predict.sh --input 0 --config configs/predict/predict.yaml  # webcam"
    exit 1
fi

if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found: $CONFIG_FILE"
    echo "Using default settings (no --config)..."
    CONFIG_FILE=""
fi

echo "=============================================="
echo "YOLO Prediction Script"
echo "=============================================="
echo "Config: ${CONFIG_FILE:-<none, using defaults>}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "=============================================="

if [ -z "$CONFIG_FILE" ]; then
    python "${SCRIPT_DIR}/yolo.py" predict "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
else
    python "${SCRIPT_DIR}/yolo.py" predict --config "$CONFIG_FILE" "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
fi