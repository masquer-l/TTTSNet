#!/usr/bin/env bash
set -euo pipefail

# TTTSNet temporal no-loss diagnostic: same dataset as temporal v1, but temporal loss weight=0
# Usage: ./scripts/train_temporal_no_loss.sh [config_path]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${1:-$PROJECT_DIR/configs/config_temporal_no_loss.json}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="tttsnet_temporal_no_loss_${TIMESTAMP}"
EXP_DIR="$PROJECT_DIR/experiments/$RUN_NAME"

mkdir -p "$EXP_DIR"
cp "$CONFIG_PATH" "$EXP_DIR/config.json"

echo "========================================"
echo "TTTSNet Temporal No-Loss Diagnostic"
echo "Config:  $CONFIG_PATH"
echo "Exp Dir: $EXP_DIR"
echo "========================================"

python3 "$PROJECT_DIR/tools/train_temporal.py" \
  --config "$CONFIG_PATH" \
  --work_dir "$EXP_DIR" \
  2>&1 | tee "$EXP_DIR/training.log"

echo "Training log saved to: $EXP_DIR/training.log"
