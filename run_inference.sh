#!/bin/bash
#
# YOLO Inference Shell Script
# ============================
# A convenient wrapper for running YOLO inference with various configurations.
#
# Usage:
#   ./run_inference.sh --model best.pt --input images/ --output results/
#   ./run_inference.sh --config configs/inference.yaml
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default config file
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/inference.yaml"

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

# Check if config file is specified
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    echo "Using config: $CONFIG_FILE"
    python "${SCRIPT_DIR}/inference.py" --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
elif [ -f "$DEFAULT_CONFIG" ]; then
    echo "Using default config: $DEFAULT_CONFIG"
    python "${SCRIPT_DIR}/inference.py" --config "$DEFAULT_CONFIG" "${EXTRA_ARGS[@]}"
else
    # Run without config file (CLI only)
    python "${SCRIPT_DIR}/inference.py" "${EXTRA_ARGS[@]}"
fi
