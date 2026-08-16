#!/usr/bin/env bash
# Launch X-AnyLabeling on the current annotation working set.
# This loads only unfinished frames (reviewed/unreviewable frames are filtered out).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BATCH_DIR="${BATCH_DIR:-/mnt/d/torch_project/dataset/sfy_screening/annotations/batch_001}"
XANYLABELING_DIR="${XANYLABELING_DIR:-/mnt/d/torch_project/X-AnyLabeling}"

WORKING_IMAGES_DIR="${BATCH_DIR}/working/images"
WORKING_LABELS_DIR="${BATCH_DIR}/working/labels"

if [ ! -d "${WORKING_IMAGES_DIR}" ]; then
    echo "Working set not found: ${WORKING_IMAGES_DIR}"
    echo "Run: python ${PROJECT_ROOT}/tools/screening/filter_working_set.py"
    exit 1
fi

cd "${XANYLABELING_DIR}"

conda activate x-anylabeling 2>/dev/null || true

export PYTHONPATH="${XANYLABELING_DIR}"

# Fix QStandardPaths runtime directory permissions warning
export XDG_RUNTIME_DIR="/tmp/runtime-$(id -u)"
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 0700 "${XDG_RUNTIME_DIR}"

python anylabeling/app.py \
    "${WORKING_IMAGES_DIR}" \
    --output "${WORKING_LABELS_DIR}" \
    --labels vessel \
    --flags reviewed,unreviewable \
    --autosave
