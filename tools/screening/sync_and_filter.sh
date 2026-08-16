#!/usr/bin/env bash
# Sync X-AnyLabeling flags back to CSV/DB and regenerate the working subset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BATCH_DIR="${BATCH_DIR:-/mnt/d/torch_project/dataset/sfy_screening/annotations/batch_001}"

echo "==> Syncing X-AnyLabeling flags to CSV and review.db"
python "${PROJECT_ROOT}/tools/screening/sync_batch_labels.py" \
    --batch-dir "${BATCH_DIR}"

echo ""
echo "==> Regenerating working subset (excluding reviewed/unreviewable frames)"
python "${PROJECT_ROOT}/tools/screening/filter_working_set.py" \
    --batch-dir "${BATCH_DIR}"

echo ""
echo "Done. Relaunch X-AnyLabeling with: ./scripts/launch_xanylabeling.sh"
