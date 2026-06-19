#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${1:-$PROJECT_DIR/config_temporal_v2.json}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="tttsnet_temporal_v2_${TIMESTAMP}"
EXP_DIR="$PROJECT_DIR/experiments/$RUN_NAME"

mkdir -p "$EXP_DIR"
cp "$CONFIG_PATH" "$EXP_DIR/config.json"

echo "========================================"
echo "TTTSNet Temporal Consistency v2 (lambda=1.0)"
echo "Config:  $CONFIG_PATH"
echo "Exp Dir: $EXP_DIR"
echo "========================================"

python3 "$PROJECT_DIR/train_temporal.py" \
  --config "$CONFIG_PATH" \
  --work_dir "$EXP_DIR" \
  2>&1 | tee "$EXP_DIR/training.log"

echo "Training log saved to: $EXP_DIR/training.log"
