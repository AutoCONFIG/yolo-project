#!/bin/bash
#
# YOLO Validation Shell Script
# =============================
# A convenient wrapper for running YOLO validation with various configurations.
#
# Usage:
#   ./run_val.sh                              # Validate with default config
#   ./run_val.sh --data coco8.yaml            # Override dataset
#   ./run_val.sh --config configs/validate/example/detect_example.yaml
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CONFIG="${SCRIPT_DIR}/configs/validate/val.yaml"

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

# Use default config if none specified
if [ -z "$CONFIG_FILE" ]; then
    if [ -f "$DEFAULT_CONFIG" ]; then
        CONFIG_FILE="$DEFAULT_CONFIG"
    fi
fi

if [ -n "$CONFIG_FILE" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo "Warning: Config file not found: $CONFIG_FILE"
    echo "Using default settings (no --config)..."
    CONFIG_FILE=""
fi

echo "=============================================="
echo "YOLO Validation Script"
echo "=============================================="
echo "Config: ${CONFIG_FILE:-<none, using defaults>}"
echo "Extra args: ${EXTRA_ARGS[*]}"
echo "=============================================="

if [ -z "$CONFIG_FILE" ]; then
    python "${SCRIPT_DIR}/yolo.py" val "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
else
    python "${SCRIPT_DIR}/yolo.py" val --config "$CONFIG_FILE" "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
fi