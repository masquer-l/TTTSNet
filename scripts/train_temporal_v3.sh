#!/usr/bin/env bash
set -euo pipefail

# TTTSNet temporal consistency training wrapper (v3 strong augmentation)
# Usage: ./scripts/train_temporal_v3.sh [config_path]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="${1:-$PROJECT_DIR/configs/config_temporal_v3.json}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="tttsnet_temporal_v3_${TIMESTAMP}"
EXP_DIR="$PROJECT_DIR/experiments/$RUN_NAME"

mkdir -p "$PROJECT_DIR/experiments"

echo "========================================"
echo "TTTSNet Temporal Consistency v3 (strong aug, lambda=0.1)"
echo "Config:  $CONFIG_PATH"
echo "Exp Dir: $EXP_DIR"
echo "========================================"

python3 "$PROJECT_DIR/tools/train_temporal.py" \
  --config "$CONFIG_PATH" \
  --work_dir "$PROJECT_DIR/experiments"

echo "Training log saved to: $EXP_DIR/training.log"
