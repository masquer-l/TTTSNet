"""SQLite database layer for the screening / annotation pipeline.

All primary state is stored in review.db. CSV snapshots can be exported on demand.
"""

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from tools.screening.config import DB_PATH, TEST_DB_PATH, UNIFIED_DIR, ensure_dirs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a database connection with production-safe PRAGMAs.

    Args:
        db_path: Optional database path. Defaults to config.DB_PATH.
    """
    conn = sqlite3.connect(db_path if db_path is not None else DB_PATH)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    ensure_dirs()
    with connect_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                `case` TEXT NOT NULL,
                video_name TEXT NOT NULL,
                raw_path TEXT,
                unified_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                fps REAL,
                frame_count INTEGER,
                duration_sec REAL,
                duration_min REAL,
                size_mb REAL,
                bitrate_mbps REAL,
                fourcc TEXT,
                status TEXT DEFAULT 'ok',
                note TEXT,
                review_status TEXT DEFAULT 'pending',
                crop_center_x REAL,
                crop_center_y REAL,
                crop_size REAL,
                PRIMARY KEY (`case`, video_name)
            );

            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                `case` TEXT NOT NULL,
                video_name TEXT NOT NULL,
                video_path TEXT NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                duration_sec REAL NOT NULL,
                preview_count INTEGER NOT NULL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                invalid_reason TEXT,
                tags TEXT,
                reviewer TEXT,
                session_id TEXT,
                updated_at TEXT,
                notes TEXT,
                parent_segment_id TEXT,
                FOREIGN KEY (`case`, video_name) REFERENCES videos(`case`, video_name)
            );

            CREATE INDEX IF NOT EXISTS idx_segments_case ON segments(`case`);
            CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_name);
            CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);

            CREATE TABLE IF NOT EXISTS segment_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL,
                label TEXT NOT NULL,
                invalid_reason TEXT,
                tags TEXT,
                reviewer TEXT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_segment_labels_segment ON segment_labels(segment_id);

            CREATE TABLE IF NOT EXISTS frames (
                segment_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                frame_time_sec REAL NOT NULL,
                is_preview INTEGER NOT NULL DEFAULT 0,
                preview_path TEXT,
                overlay_path TEXT,
                fullres_path TEXT,
                annotation_mask_path TEXT,
                cnn_max_conf REAL,
                cnn_mean_conf REAL,
                cnn_area_ratio REAL,
                status TEXT DEFAULT 'pending',
                tags TEXT,
                label_source TEXT DEFAULT 'screening',
                PRIMARY KEY (segment_id, frame_idx),
                FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_frames_preview ON frames(segment_id, is_preview);

            CREATE TABLE IF NOT EXISTS frame_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                label TEXT NOT NULL,
                invalid_reason TEXT,
                reviewer TEXT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                notes TEXT,
                tags TEXT,
                annotation_json_path TEXT,
                annotation_mask_path TEXT,
                label_source TEXT DEFAULT 'screening',
                FOREIGN KEY (segment_id, frame_idx) REFERENCES frames(segment_id, frame_idx)
            );

            CREATE INDEX IF NOT EXISTS idx_frame_labels_frame ON frame_labels(segment_id, frame_idx);

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                reviewer TEXT NOT NULL,
                action TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                segments_reviewed INTEGER DEFAULT 0,
                frames_reviewed INTEGER DEFAULT 0,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT
            );
            """
        )
        conn.execute("PRAGMA journal_mode=WAL")
        _apply_migrations(conn)


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply idempotent schema migrations and record them in schema_version."""
    migrations = [
        (1, "Add annotation_mask_path to frames", "ALTER TABLE frames ADD COLUMN annotation_mask_path TEXT"),
        (2, "Add crop columns to videos", [
            "ALTER TABLE videos ADD COLUMN crop_center_x REAL",
            "ALTER TABLE videos ADD COLUMN crop_center_y REAL",
            "ALTER TABLE videos ADD COLUMN crop_size REAL",
        ]),
        (3, "Add tags to frames", "ALTER TABLE frames ADD COLUMN tags TEXT"),
        (4, "Add tags to frame_labels", "ALTER TABLE frame_labels ADD COLUMN tags TEXT"),
        (5, "Add label_source to frames", "ALTER TABLE frames ADD COLUMN label_source TEXT DEFAULT 'screening'"),
        (6, "Add label_source to frame_labels", "ALTER TABLE frame_labels ADD COLUMN label_source TEXT DEFAULT 'screening'"),
    ]

    def _record(version: int, description: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
            (version, _now(), description),
        )

    for version, description, statements in migrations:
        already_applied = conn.execute(
            "SELECT 1 FROM schema_version WHERE version = ?", (version,)
        ).fetchone()
        if already_applied:
            continue
        if isinstance(statements, str):
            statements = [statements]
        for sql in statements:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                # Column may already exist from legacy init; treat as applied.
                if "duplicate column" in str(e).lower():
                    pass
                else:
                    raise
        _record(version, description)


def init_test_db() -> None:
    """Initialize the isolated test database schema (no data).

    Use this for automated tests and CI. Production review.db must never be
    touched by test code.
    """
    from tools.screening.config import TEST_DB_PATH, ensure_dirs

    ensure_dirs()
    with connect_db(TEST_DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS videos (
                `case` TEXT NOT NULL,
                video_name TEXT NOT NULL,
                raw_path TEXT,
                unified_path TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                fps REAL,
                frame_count INTEGER,
                duration_sec REAL,
                duration_min REAL,
                size_mb REAL,
                bitrate_mbps REAL,
                fourcc TEXT,
                status TEXT DEFAULT 'ok',
                note TEXT,
                review_status TEXT DEFAULT 'pending',
                crop_center_x REAL,
                crop_center_y REAL,
                crop_size REAL,
                PRIMARY KEY (`case`, video_name)
            );

            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                `case` TEXT NOT NULL,
                video_name TEXT NOT NULL,
                video_path TEXT NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                duration_sec REAL NOT NULL,
                preview_count INTEGER NOT NULL DEFAULT 0,
                status TEXT DEFAULT 'pending',
                invalid_reason TEXT,
                tags TEXT,
                reviewer TEXT,
                session_id TEXT,
                updated_at TEXT,
                notes TEXT,
                parent_segment_id TEXT,
                FOREIGN KEY (`case`, video_name) REFERENCES videos(`case`, video_name)
            );

            CREATE INDEX IF NOT EXISTS idx_segments_case ON segments(`case`);
            CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_name);
            CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);

            CREATE TABLE IF NOT EXISTS segment_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL,
                label TEXT NOT NULL,
                invalid_reason TEXT,
                tags TEXT,
                reviewer TEXT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_segment_labels_segment ON segment_labels(segment_id);

            CREATE TABLE IF NOT EXISTS frames (
                segment_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                frame_time_sec REAL NOT NULL,
                is_preview INTEGER NOT NULL DEFAULT 0,
                preview_path TEXT,
                overlay_path TEXT,
                fullres_path TEXT,
                annotation_mask_path TEXT,
                cnn_max_conf REAL,
                cnn_mean_conf REAL,
                cnn_area_ratio REAL,
                status TEXT DEFAULT 'pending',
                tags TEXT,
                label_source TEXT DEFAULT 'screening',
                PRIMARY KEY (segment_id, frame_idx),
                FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_frames_preview ON frames(segment_id, is_preview);

            CREATE TABLE IF NOT EXISTS frame_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                label TEXT NOT NULL,
                invalid_reason TEXT,
                reviewer TEXT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                notes TEXT,
                tags TEXT,
                annotation_json_path TEXT,
                annotation_mask_path TEXT,
                label_source TEXT DEFAULT 'screening',
                FOREIGN KEY (segment_id, frame_idx) REFERENCES frames(segment_id, frame_idx)
            );

            CREATE INDEX IF NOT EXISTS idx_frame_labels_frame ON frame_labels(segment_id, frame_idx);

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                reviewer TEXT NOT NULL,
                action TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                segments_reviewed INTEGER DEFAULT 0,
                frames_reviewed INTEGER DEFAULT 0,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT
            );
            """
        )
        conn.execute("PRAGMA journal_mode=WAL")
        _apply_migrations(conn)


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------

def load_videos_from_metadata(metadata_csv: Path) -> int:
    """Populate the videos table from the existing metadata CSV."""
    count = 0
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        existing = {
            (r["case"], r["video_name"])
            for r in conn.execute("SELECT `case`, video_name FROM videos")
        }
    with open(metadata_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            key = (row["case"], row["video_name"])
            if key in existing:
                continue
            unified = str(UNIFIED_DIR / row["case"] / row["video_name"])
            rows.append(
                (
                    row["case"],
                    row["video_name"],
                    None,
                    unified,
                    _int(row.get("width")),
                    _int(row.get("height")),
                    _float(row.get("fps")),
                    _int(row.get("frame_count")),
                    _float(row.get("duration_sec")),
                    _float(row.get("duration_min")),
                    _float(row.get("size_mb")),
                    _float(row.get("bitrate_mbps")),
                    row.get("fourcc"),
                    row.get("status", "ok") or "ok",
                    row.get("note"),
                    "pending",
                )
            )
    if not rows:
        return 0
    with connect_db() as conn:
        conn.executemany(
            """
            INSERT INTO videos
            (`case`, video_name, raw_path, unified_path, width, height, fps,
             frame_count, duration_sec, duration_min, size_mb, bitrate_mbps,
             fourcc, status, note, review_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        count = len(rows)
    return count


def update_video_crop(
    case: str,
    video_name: str,
    crop_center_x: float,
    crop_center_y: float,
    crop_size: float,
) -> None:
    init_db()
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE videos
            SET crop_center_x = ?, crop_center_y = ?, crop_size = ?
            WHERE `case` = ? AND video_name = ?
            """,
            (crop_center_x, crop_center_y, crop_size, case, video_name),
        )


def get_video_crop(case: str, video_name: str) -> Optional[Dict[str, float]]:
    init_db()
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT crop_center_x, crop_center_y, crop_size FROM videos WHERE `case` = ? AND video_name = ?",
            (case, video_name),
        ).fetchone()
    if row is None or row["crop_size"] is None:
        return None
    return {
        "center_x": float(row["crop_center_x"]),
        "center_y": float(row["crop_center_y"]),
        "crop_size": float(row["crop_size"]),
    }


def invalidate_video(
    case: str,
    video_name: str,
    invalid_reason: Optional[str] = None,
    notes: Optional[str] = None,
    reviewer: str = "default",
    session_id: Optional[str] = None,
) -> Tuple[int, int]:
    """Mark an entire video as invalid and cascade to all its segments/frames.

    Returns:
        (number of segments invalidated, number of frames invalidated)
    """
    init_db()
    ts = _now()
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE videos
            SET review_status = 'invalid', note = COALESCE(note || '; ', '') || ?
            WHERE `case` = ? AND video_name = ?
            """,
            (notes or f"Video invalidated: {invalid_reason or 'not specified'}", case, video_name),
        )

        # Invalidate all segments and record audit.
        segments = conn.execute(
            "SELECT segment_id FROM segments WHERE `case` = ? AND video_name = ?",
            (case, video_name),
        ).fetchall()
        for (segment_id,) in segments:
            conn.execute(
                """
                UPDATE segments
                SET status = 'invalid', invalid_reason = ?, reviewer = ?, session_id = ?, updated_at = ?, notes = ?
                WHERE segment_id = ?
                """,
                (invalid_reason, reviewer, session_id, ts, notes, segment_id),
            )
            conn.execute(
                """
                INSERT INTO segment_labels
                (segment_id, label, invalid_reason, reviewer, session_id, timestamp, notes)
                VALUES (?, 'invalid', ?, ?, ?, ?, ?)
                """,
                (segment_id, invalid_reason, reviewer, session_id, ts, notes),
            )

        # Invalidate all frames and record audit.
        frames = conn.execute(
            "SELECT segment_id, frame_idx FROM frames WHERE segment_id IN (SELECT segment_id FROM segments WHERE `case` = ? AND video_name = ?)",
            (case, video_name),
        ).fetchall()
        for segment_id, frame_idx in frames:
            conn.execute(
                """
                UPDATE frames
                SET status = 'invalid'
                WHERE segment_id = ? AND frame_idx = ?
                """,
                (segment_id, frame_idx),
            )
            conn.execute(
                """
                INSERT INTO frame_labels
                (segment_id, frame_idx, label, invalid_reason, reviewer, session_id, timestamp, notes, label_source)
                VALUES (?, ?, 'invalid', ?, ?, ?, ?, ?, 'screening')
                """,
                (segment_id, frame_idx, invalid_reason, reviewer, session_id, ts, notes),
            )

        return len(segments), len(frames)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def build_registry(segment_sec: Optional[float] = None) -> int:
    """Create segments and frames rows for all ok videos."""
    from tools.screening.config import DEFAULT_SEGMENT_SEC, PREVIEW_INTERVAL_SEC

    init_db()
    segment_sec = segment_sec or DEFAULT_SEGMENT_SEC
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        videos = [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM videos WHERE status = 'ok' ORDER BY `case`, video_name"
            )
        ]

    inserted_segments = 0
    inserted_frames = 0
    for v in videos:
        video_path = Path(v["unified_path"])
        if not video_path.exists():
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or total_frames <= 0:
            continue

        frames_per_segment = int(round(segment_sec * fps))
        preview_step = int(round(PREVIEW_INTERVAL_SEC * fps))
        case = v["case"]
        video_name = v["video_name"]
        video_stem = Path(video_name).stem

        seg_rows = []
        frame_rows = []
        start = 0
        while start < total_frames:
            end = min(start + frames_per_segment - 1, total_frames - 1)
            seg_id = f"{case}_{video_stem}_{start:08d}_{end:08d}"
            duration = (end - start + 1) / fps
            preview_indices = set(
                range(start, end + 1, preview_step)
            )
            seg_rows.append(
                (
                    seg_id,
                    case,
                    video_name,
                    str(video_path),
                    start,
                    end,
                    end - start + 1,
                    duration,
                    len(preview_indices),
                )
            )
            for fidx in range(start, end + 1):
                frame_rows.append(
                    (
                        seg_id,
                        fidx,
                        fidx / fps,
                        1 if fidx in preview_indices else 0,
                    )
                )
            start = end + 1

        with connect_db() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO segments
                (segment_id, `case`, video_name, video_path, start_frame, end_frame,
                 frame_count, duration_sec, preview_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                seg_rows,
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO frames
                (segment_id, frame_idx, frame_time_sec, is_preview)
                VALUES (?, ?, ?, ?)
                """,
                frame_rows,
            )
            inserted_segments += conn.total_changes - inserted_frames  # approximate
            inserted_frames += len(frame_rows)
    return len(videos)


def update_segment_status(
    segment_id: str,
    status: str,
    invalid_reason: Optional[str] = None,
    tags: Optional[str] = None,
    notes: Optional[str] = None,
    reviewer: str = "default",
    session_id: Optional[str] = None,
) -> None:
    init_db()
    ts = _now()
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE segments
            SET status = ?, invalid_reason = ?, tags = ?, reviewer = ?, session_id = ?, updated_at = ?, notes = ?
            WHERE segment_id = ?
            """,
            (status, invalid_reason, tags, reviewer, session_id, ts, notes, segment_id),
        )
        conn.execute(
            """
            INSERT INTO segment_labels
            (segment_id, label, invalid_reason, tags, reviewer, session_id, timestamp, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (segment_id, status, invalid_reason, tags, reviewer, session_id, ts, notes),
        )


def split_segment(segment_id: str, split_frame_idx: int, reason: Optional[str] = None) -> Optional[Tuple[str, str]]:
    init_db()
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        if row is None:
            return None
        start = row["start_frame"]
        end = row["end_frame"]
        if split_frame_idx <= start or split_frame_idx >= end:
            return None
        case = row["case"]
        video_name = row["video_name"]
        video_path = row["video_path"]
        video_stem = Path(video_name).stem

        left_id = f"{case}_{video_stem}_{start:08d}_{split_frame_idx - 1:08d}"
        right_id = f"{case}_{video_stem}_{split_frame_idx:08d}_{end:08d}"

        # mark parent split
        conn.execute(
            "UPDATE segments SET status = 'split', notes = ? WHERE segment_id = ?",
            (reason or "split", segment_id),
        )

        # create children (frames will be regenerated on next build, or we copy)
        # Simpler: delete old frames and insert children frames
        conn.execute("DELETE FROM frames WHERE segment_id = ?", (segment_id,))

        fps = (end - start + 1) / row["duration_sec"]
        from tools.screening.config import PREVIEW_INTERVAL_SEC

        preview_step = int(round(PREVIEW_INTERVAL_SEC * fps))

        def _add_seg(seg_id, s, e):
            duration = (e - s + 1) / fps
            preview_indices = set(range(s, e + 1, preview_step))
            conn.execute(
                """
                INSERT OR IGNORE INTO segments
                (segment_id, `case`, video_name, video_path, start_frame, end_frame,
                 frame_count, duration_sec, preview_count, parent_segment_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seg_id,
                    case,
                    video_name,
                    video_path,
                    s,
                    e,
                    e - s + 1,
                    duration,
                    len(preview_indices),
                    segment_id,
                ),
            )
            for fidx in range(s, e + 1):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO frames
                    (segment_id, frame_idx, frame_time_sec, is_preview)
                    VALUES (?, ?, ?, ?)
                    """,
                    (seg_id, fidx, fidx / fps, 1 if fidx in preview_indices else 0),
                )

        _add_seg(left_id, start, split_frame_idx - 1)
        _add_seg(right_id, split_frame_idx, end)
    return left_id, right_id


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def update_frame_label(
    segment_id: str,
    frame_idx: int,
    label: str,
    invalid_reason: Optional[str] = None,
    tags: Optional[str] = None,
    notes: Optional[str] = None,
    reviewer: str = "default",
    session_id: Optional[str] = None,
    annotation_json_path: Optional[str] = None,
    annotation_mask_path: Optional[str] = None,
    label_source: Optional[str] = None,
) -> None:
    init_db()
    ts = _now()
    # Infer label_source: a frame with a pixel mask is manual GT; otherwise it is
    # a screening decision without pixel annotation.
    if label_source is None:
        label_source = "manual" if annotation_mask_path is not None else "screening"
    update_fields = ["status = ?", "tags = ?", "label_source = ?"]
    update_values = [label, tags, label_source]
    if annotation_mask_path is not None:
        update_fields.append("annotation_mask_path = ?")
        update_values.append(annotation_mask_path)
    update_values.extend([segment_id, frame_idx])
    with connect_db() as conn:
        conn.execute(
            f"UPDATE frames SET {', '.join(update_fields)} WHERE segment_id = ? AND frame_idx = ?",
            update_values,
        )
        conn.execute(
            """
            INSERT INTO frame_labels
            (segment_id, frame_idx, label, invalid_reason, tags, reviewer, session_id,
             timestamp, notes, annotation_json_path, annotation_mask_path, label_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                segment_id,
                frame_idx,
                label,
                invalid_reason,
                tags,
                reviewer,
                session_id,
                ts,
                notes,
                annotation_json_path,
                annotation_mask_path,
                label_source,
            ),
        )


def set_frame_paths(
    segment_id: str,
    frame_idx: int,
    preview_path: Optional[str] = None,
    overlay_path: Optional[str] = None,
    fullres_path: Optional[str] = None,
    cnn_max_conf: Optional[float] = None,
    cnn_mean_conf: Optional[float] = None,
    cnn_area_ratio: Optional[float] = None,
) -> None:
    init_db()
    fields = []
    vals = []
    for name, val in [
        ("preview_path", preview_path),
        ("overlay_path", overlay_path),
        ("fullres_path", fullres_path),
        ("cnn_max_conf", cnn_max_conf),
        ("cnn_mean_conf", cnn_mean_conf),
        ("cnn_area_ratio", cnn_area_ratio),
    ]:
        if val is not None:
            fields.append(f"{name} = ?")
            vals.append(val)
    if not fields:
        return
    vals.extend([segment_id, frame_idx])
    with connect_db() as conn:
        conn.execute(
            f"UPDATE frames SET {', '.join(fields)} WHERE segment_id = ? AND frame_idx = ?",
            vals,
        )


def set_frame_paths_batch(
    updates: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[float], Optional[float], Optional[float], str, int]]
) -> None:
    """Batch update frame paths and CNN stats.

    Each tuple: (preview_path, overlay_path, fullres_path, cnn_max_conf,
                 cnn_mean_conf, cnn_area_ratio, segment_id, frame_idx)

    ``None`` values preserve the existing column value; use ``set_frame_paths``
    with an explicit empty string if a field must be cleared.
    """
    init_db()
    with connect_db() as conn:
        conn.executemany(
            """
            UPDATE frames
            SET preview_path = COALESCE(?, preview_path),
                overlay_path = COALESCE(?, overlay_path),
                fullres_path = COALESCE(?, fullres_path),
                cnn_max_conf = COALESCE(?, cnn_max_conf),
                cnn_mean_conf = COALESCE(?, cnn_mean_conf),
                cnn_area_ratio = COALESCE(?, cnn_area_ratio)
            WHERE segment_id = ? AND frame_idx = ?
            """,
            updates,
        )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def start_session(action: str, reviewer: str = "default", notes: Optional[str] = None) -> str:
    init_db()
    ts = _now()
    session_id = f"{reviewer}_{action}_{ts.replace(':', '-')}"
    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, reviewer, action, started_at, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, reviewer, action, ts, notes),
        )
    return session_id


def end_session(session_id: str, segments_reviewed: int = 0, frames_reviewed: int = 0) -> None:
    with connect_db() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET ended_at = ?, segments_reviewed = ?, frames_reviewed = ?
            WHERE session_id = ?
            """,
            (_now(), segments_reviewed, frames_reviewed, session_id),
        )


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_summary() -> Dict[str, Any]:
    init_db()
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        total_videos = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE status = 'ok'"
        ).fetchone()[0]
        total_segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        status_counts = {
            r["status"]: r["n"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM segments GROUP BY status"
            )
        }
        total_frames = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        frame_status_counts = {
            r["status"]: r["n"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM frames GROUP BY status"
            )
        }
    return {
        "videos_ok": total_videos,
        "segments": total_segments,
        "segment_status": status_counts,
        "frames": total_frames,
        "frame_status": frame_status_counts,
    }


def get_segments(
    status: Optional[str] = None,
    case: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    sort: str = "segment_id",
) -> List[Dict[str, Any]]:
    init_db()
    where = ["1=1"]
    params: List[Any] = []
    if status:
        where.append("status = ?")
        params.append(status)
    if case:
        where.append("`case` = ?")
        params.append(case)
    sql = f"SELECT * FROM segments WHERE {' AND '.join(where)} ORDER BY {sort} LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


def get_segment(segment_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        return dict(row) if row else None


def get_frames(segment_id: str, only_preview: bool = False) -> List[Dict[str, Any]]:
    init_db()
    sql = "SELECT * FROM frames WHERE segment_id = ?"
    params: List[Any] = [segment_id]
    if only_preview:
        sql += " AND is_preview = 1"
    sql += " ORDER BY frame_idx"
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params)]


# ---------------------------------------------------------------------------
# Snapshots / export helpers
# ---------------------------------------------------------------------------

def export_snapshots(snapshot_dir: Path) -> None:
    init_db()
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tables = ["videos", "segments", "segment_labels", "frames", "frame_labels", "sessions"]
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        for table in tables:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            out = snapshot_dir / f"{table}.csv"
            with open(out, "w", encoding="utf-8", newline="") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys()])
                    writer.writeheader()
                    writer.writerows([dict(r) for r in rows])
                else:
                    f.write("")


def _int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None and v != "" else None
    except ValueError:
        return None


def _float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "" else None
    except ValueError:
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize screening database schema.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Initialize the isolated test database (review_test.db) instead of production review.db.",
    )
    args = parser.parse_args()

    if args.test:
        init_test_db()
        print(f"Test database initialized: {TEST_DB_PATH}")
    else:
        init_db()
        print(f"Production database initialized: {DB_PATH}")
