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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CONFIG="${SCRIPT_DIR}/configs/default.yaml"

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

if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="$DEFAULT_CONFIG"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found: $CONFIG_FILE"
    echo "Using default settings (no --config)..."
    CONFIG_FILE=""
fi

echo "=============================================="
echo "YOLO Training Script"
echo "=============================================="
echo "Config: ${CONFIG_FILE:-<none, using pure defaults>}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "=============================================="

if [ -z "$CONFIG_FILE" ]; then
    python "${SCRIPT_DIR}/train.py" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
else
    python "${SCRIPT_DIR}/train.py" --config "$CONFIG_FILE" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
fi
