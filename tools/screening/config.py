"""Central configuration for the SFY video screening / annotation pipeline."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path("/mnt/d/torch_project/TTTSNet")
DATASET_ROOT = Path("/mnt/d/torch_project/dataset")

# Source videos (hard-linked unified directory)
UNIFIED_DIR = DATASET_ROOT / "sfy_source_unified"

# Screening outputs
SCREENING_DIR = DATASET_ROOT / "sfy_screening"
PREVIEW_DIR = SCREENING_DIR / "previews"
OVERLAY_DIR = SCREENING_DIR / "overlays"
SEGMENT_FRAMES_DIR = SCREENING_DIR / "segment_frames"
ANNOTATIONS_DIR = SCREENING_DIR / "annotations"
EXPORTS_DIR = SCREENING_DIR / "exports"
DB_DIR = SCREENING_DIR / "db"
LOGS_DIR = SCREENING_DIR / "logs"

# Source metadata
METADATA_CSV = PROJECT_ROOT / "experiments" / "sfy_video_metadata.csv"

# Previews
PREVIEW_INTERVAL_SEC = 2
PREVIEW_MAX_SIZE = 960          # max edge, preserve aspect ratio
PREVIEW_QUALITY = 80

# Segments
DEFAULT_SEGMENT_SEC = 30        # default segment length

# CNN baseline for overlays
CNN_CONFIG = PROJECT_ROOT / "configs" / "config_local.json"
CNN_CHECKPOINT = PROJECT_ROOT / "TTTSNet_best_model.pth"
CNN_IMG_SIZE = 448
CNN_BATCH_SIZE = 16

# DB
# Use SFY_USE_TEST_DB=1 to connect to the isolated test database instead of
# the production review.db. All automated tests MUST use the test database.
if os.environ.get("SFY_USE_TEST_DB", "").lower() in ("1", "true", "yes"):
    DB_PATH = DB_DIR / "review_test.db"
else:
    DB_PATH = DB_DIR / "review.db"

# Test DB path (exposed for explicit initialization)
TEST_DB_PATH = DB_DIR / "review_test.db"
PROD_DB_PATH = DB_DIR / "review.db"

# Default reviewer / session
DEFAULT_REVIEWER = "default"

# Label vocabularies
SEGMENT_STATUSES = ["pending", "valid", "invalid", "uncertain", "split"]
FRAME_STATUSES = ["pending", "valid", "invalid", "uncertain"]

INVALID_REASONS = [
    "no_vessel",
    "out_of_focus",
    "overexposure",
    "underexposure",
    "laser_glare",
    "instrument_occlusion",
    "blood_debris",
    "motion_blur",
    "outside_body",
    "corrupted",
    "other",
]

DIFFICULTY_TAGS = [
    "fine_vessel",
    "low_contrast",
    "laser_spot",
    "instrument",
    "small_foreground",
    "large_foreground",
    "boundary_artifact",
]


def ensure_dirs() -> None:
    for p in [
        PREVIEW_DIR,
        OVERLAY_DIR,
        SEGMENT_FRAMES_DIR,
        ANNOTATIONS_DIR,
        EXPORTS_DIR,
        DB_DIR,
        LOGS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
