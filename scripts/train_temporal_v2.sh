#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${1:-$PROJECT_DIR/configs/config_temporal_v2.json}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="tttsnet_temporal_v2_${TIMESTAMP}"
EXP_DIR="$PROJECT_DIR/experiments/$RUN_NAME"

mkdir -p "$PROJECT_DIR/experiments"

echo "========================================"
echo "TTTSNet Temporal Consistency v2 (lambda=1.0)"
echo "Config:  $CONFIG_PATH"
echo "Exp Dir: $EXP_DIR"
echo "========================================"

python3 "$PROJECT_DIR/tools/train_temporal.py" \
  --config "$CONFIG_PATH" \
  --work_dir "$PROJECT_DIR/experiments"

echo "Training log saved to: $EXP_DIR/training.log"
