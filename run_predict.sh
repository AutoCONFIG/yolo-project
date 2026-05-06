#!/bin/bash
#
# YOLO Prediction Shell Script
# ==============================
# A convenient wrapper for running YOLO inference.
#
# Usage:
#   ./run_predict.sh --config configs/predict/example/detect_example.yaml
#   ./run_predict.sh --config configs/predict/chaoyuan.yaml --input image.jpg
#   ./run_predict.sh                                     # Use default config
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default config file
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/predict/example/detect_example.yaml"

# Parse arguments
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

# Determine which config to use
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Using config: $CONFIG_FILE"
    python "${SCRIPT_DIR}/yolo.py" predict --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
elif [ -f "$DEFAULT_CONFIG" ]; then
    echo "Using default config: $DEFAULT_CONFIG"
    python "${SCRIPT_DIR}/yolo.py" predict --config "$DEFAULT_CONFIG" "${EXTRA_ARGS[@]}"
else
    echo "No config file found. Using CLI arguments only."
    python "${SCRIPT_DIR}/yolo.py" predict "${EXTRA_ARGS[@]}"
fi
