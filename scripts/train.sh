#!/usr/bin/env bash
set -euo pipefail

# TTTSNet single-frame baseline training wrapper
# Usage: ./scripts/train.sh [config_path]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${1:-$PROJECT_DIR/configs/config.json}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="tttsnet_single_${TIMESTAMP}"
EXP_DIR="$PROJECT_DIR/experiments/$RUN_NAME"

mkdir -p "$PROJECT_DIR/experiments"

echo "========================================"
echo "TTTSNet Single-Frame Baseline"
echo "Config:  $CONFIG_PATH"
echo "Exp Dir: $EXP_DIR"
echo "========================================"

python3 "$PROJECT_DIR/tools/train.py" \
  --config "$CONFIG_PATH" \
  --work_dir "$PROJECT_DIR/experiments"

echo "Training log saved to: $EXP_DIR/training.log"
