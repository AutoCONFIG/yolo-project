#!/bin/bash
#
# YOLO Training Shell Script
# ===========================
# A convenient wrapper for running YOLO training with various configurations.
#
# Usage:
#   ./run_train.sh                          # Train with default config
#   ./run_train.sh --epochs 50              # Override epochs
#   ./run_train.sh --config my_config.yaml  # Use custom config
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default config file
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/default.yaml"

# Parse arguments
CONFIG_FILE="configs/train/xiaotu_8classes.yaml"
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

# Use default config if not specified
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found: $CONFIG_FILE"
    echo "Using default settings..."
fi

# Run training
echo "=============================================="
echo "YOLO Training Script"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "=============================================="

python "${SCRIPT_DIR}/train.py" --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
