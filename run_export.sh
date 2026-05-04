#!/bin/bash
#
# YOLO Export Shell Script
# =========================
# A convenient wrapper for exporting YOLO models to various formats.
#
# Usage:
#   ./run_export.sh --model best.pt                          # Export to ONNX
#   ./run_export.sh --model best.pt --format engine --half true  # Export to TensorRT
#   ./run_export.sh --config configs/export/onnx.yaml        # Using config file
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default config file
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/export/example/onnx/detect_example.yaml"

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
    python "${SCRIPT_DIR}/yolo.py" export --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
elif [ -f "$DEFAULT_CONFIG" ]; then
    echo "Using default config: $DEFAULT_CONFIG"
    python "${SCRIPT_DIR}/yolo.py" export --config "$DEFAULT_CONFIG" "${EXTRA_ARGS[@]}"
else
    echo "No config file found. Using CLI arguments only."
    python "${SCRIPT_DIR}/yolo.py" export "${EXTRA_ARGS[@]}"
fi