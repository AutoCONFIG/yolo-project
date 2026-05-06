#!/bin/bash
#
# YOLO Tracking Shell Script
# ===========================
# A convenient wrapper for running YOLO object tracking.
#
# Usage:
#   ./run_track.sh --config configs/predict/example/track_example.yaml
#   ./run_track.sh --model best.pt --input video.mp4 --tracker botsort.yaml
#   ./run_track.sh                                     # Use default config
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default config file
DEFAULT_CONFIG="${SCRIPT_DIR}/configs/predict/example/track_example.yaml"

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
    python "${SCRIPT_DIR}/yolo.py" track --config "$CONFIG_FILE" "${EXTRA_ARGS[@]}"
elif [ -f "$DEFAULT_CONFIG" ]; then
    echo "Using default config: $DEFAULT_CONFIG"
    python "${SCRIPT_DIR}/yolo.py" track --config "$DEFAULT_CONFIG" "${EXTRA_ARGS[@]}"
else
    echo "No config file found. Using CLI arguments only."
    python "${SCRIPT_DIR}/yolo.py" track "${EXTRA_ARGS[@]}"
fi
