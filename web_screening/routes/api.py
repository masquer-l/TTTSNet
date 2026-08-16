"""REST API routes."""

import sqlite3
from pathlib import Path
from typing import Optional

import cv2
from flask import Blueprint, Response, jsonify, request, send_file

from tools.screening.config import (
    FRAME_STATUSES,
    OVERLAY_DIR,
    PREVIEW_DIR,
    SEGMENT_FRAMES_DIR,
    SEGMENT_STATUSES,
    ensure_dirs,
)
from tools.screening.db import (
    connect_db,
    end_session,
    get_frames,
    get_segment,
    get_segments,
    get_summary,
    init_db,
    invalidate_video,
    split_segment,
    start_session,
    update_frame_label,
    update_segment_status,
)

bp = Blueprint("api", __name__)


def _case_and_stem(segment_id: str) -> tuple:
    parts = segment_id.split("_")
    return parts[0], "_".join(parts[1:-2])


def _safe_path(base_dir: Path, rel_path: str) -> Optional[Path]:
    """Resolve a relative path under base_dir, rejecting directory traversal."""
    try:
        target = (base_dir / rel_path).resolve()
        target.relative_to(base_dir.resolve())
        return target
    except (ValueError, RuntimeError):
        return None


def _serve_image(image_path: Path) -> Response:
    if not image_path.exists():
        return jsonify({"error": "image not found"}), 404
    return send_file(str(image_path), mimetype="image/jpeg", max_age=86400)


def _serve_mask(mask_path: Path) -> Response:
    if not mask_path.exists():
        return jsonify({"error": "mask not found"}), 404
    return send_file(str(mask_path), mimetype="image/png", max_age=86400)


@bp.get("/summary")
def summary():
    return jsonify(get_summary())


@bp.get("/tags")
def tags():
    """Return all tags that have been used anywhere in the project."""
    init_db()
    all_tags: set = set()
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        for table, column in [
            ("segments", "tags"),
            ("frames", "tags"),
            ("segment_labels", "tags"),
            ("frame_labels", "tags"),
        ]:
            rows = conn.execute(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL").fetchall()
            for row in rows:
                for tag in (row[column] or "").split(","):
                    tag = tag.strip()
                    if tag:
                        all_tags.add(tag)
    return jsonify(sorted(all_tags))


@bp.get("/videos")
def videos():
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT v.*,
                COUNT(s.segment_id) AS total_segments,
                SUM(CASE WHEN s.status = 'valid' THEN 1 ELSE 0 END) AS valid_segments,
                SUM(CASE WHEN s.status = 'invalid' THEN 1 ELSE 0 END) AS invalid_segments,
                SUM(CASE WHEN s.status = 'pending' THEN 1 ELSE 0 END) AS pending_segments
            FROM videos v
            LEFT JOIN segments s ON v.`case` = s.`case` AND v.video_name = s.video_name
            WHERE v.status = 'ok'
            GROUP BY v.`case`, v.video_name
            ORDER BY v.`case`, v.video_name
            """
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@bp.post("/videos/<case>/<video_name>/invalidate")
def mark_video_invalid(case: str, video_name: str):
    data = request.get_json(force=True, silent=True) or {}
    invalid_reason = data.get("invalid_reason") or "not_ttts"
    allowed_reasons = ["not_ttts", "no_vessel", "out_of_focus", "overexposure",
                       "underexposure", "laser_glare", "instrument_occlusion",
                       "blood_debris", "motion_blur", "outside_body", "corrupted", "other"]
    if invalid_reason not in allowed_reasons:
        return jsonify({"error": f"invalid reason: {invalid_reason}"}), 400

    n_segments, n_frames = invalidate_video(
        case=case,
        video_name=video_name,
        invalid_reason=invalid_reason,
        notes=data.get("notes"),
        reviewer=data.get("reviewer", "default"),
        session_id=data.get("session_id"),
    )
    return jsonify({
        "case": case,
        "video_name": video_name,
        "invalidated_segments": n_segments,
        "invalidated_frames": n_frames,
    })


@bp.get("/segments")
def segments():
    status = request.args.get("status") or None
    case = request.args.get("case") or None
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    sort = request.args.get("sort", "segment_id")
    rows = get_segments(status=status, case=case, limit=limit, offset=offset, sort=sort)
    # attach first preview path for thumbnail
    for r in rows:
        r["thumbnail_url"] = None
        r["first_preview_frame_idx"] = None
        if r.get("status") != "split":
            previews = get_frames(r["segment_id"], only_preview=True)
            if previews:
                r["thumbnail_url"] = f"/api/preview/{previews[0]['preview_path']}"
                r["first_preview_frame_idx"] = previews[0]["frame_idx"]
    return jsonify(rows)


@bp.get("/segments/<segment_id>")
def segment_detail(segment_id: str):
    seg = get_segment(segment_id)
    if not seg:
        return jsonify({"error": "not found"}), 404
    seg["frames"] = get_frames(segment_id)
    return jsonify(seg)


@bp.post("/segments/<segment_id>/label")
def label_segment(segment_id: str):
    data = request.get_json(force=True, silent=True) or {}
    status = data.get("label", "uncertain")
    if status not in SEGMENT_STATUSES:
        return jsonify({"error": f"invalid label: {status}"}), 400
    update_segment_status(
        segment_id=segment_id,
        status=status,
        invalid_reason=data.get("invalid_reason"),
        tags=data.get("tags"),
        notes=data.get("notes"),
        reviewer=data.get("reviewer", "default"),
        session_id=data.get("session_id"),
    )
    return jsonify(get_segment(segment_id))


@bp.post("/segments/<segment_id>/split")
def split(segment_id: str):
    data = request.get_json(force=True, silent=True) or {}
    split_frame = data.get("split_frame_idx")
    if split_frame is None:
        return jsonify({"error": "split_frame_idx required"}), 400
    result = split_segment(segment_id, int(split_frame), reason=data.get("reason"))
    if result is None:
        return jsonify({"error": "invalid split point"}), 400
    return jsonify({"left": result[0], "right": result[1]})


@bp.post("/segments/<segment_id>/extract")
def extract_segment(segment_id: str):
    from tools.screening.extract_segment_frames import extract_segment_frames

    ensure_dirs()
    count = extract_segment_frames(segment_id)
    return jsonify({"segment_id": segment_id, "extracted_frames": count})


@bp.get("/frames")
def frames():
    segment_id = request.args.get("segment_id") or None
    case = request.args.get("case") or None
    video_name = request.args.get("video_name") or None
    status = request.args.get("status") or None
    only_preview = request.args.get("only_preview", "1") == "1"
    limit = request.args.get("limit", 48, type=int)
    offset = request.args.get("offset", 0, type=int)

    if segment_id:
        rows = get_frames(segment_id, only_preview=only_preview)
        for r in rows:
            if r.get("preview_path"):
                r["preview_url"] = f"/api/preview/{r['preview_path']}"
            if r.get("overlay_path"):
                r["overlay_url"] = f"/api/overlay/{r['overlay_path']}"
        return jsonify({"frames": rows, "total": len(rows)})

    where = ["1=1"]
    params: list = []
    if case:
        where.append("s.`case` = ?")
        params.append(case)
    if video_name:
        where.append("s.video_name = ?")
        params.append(video_name)
    if status:
        where.append("f.status = ?")
        params.append(status)
    if only_preview:
        where.append("f.is_preview = 1")

    sort = request.args.get("sort", "frame_idx_asc")
    allowed_sort = {
        "frame_idx_asc": "s.`case`, s.video_name, f.frame_idx",
        "frame_idx_desc": "s.`case`, s.video_name, f.frame_idx DESC",
        "cnn_ratio_desc": "f.cnn_area_ratio IS NULL, f.cnn_area_ratio DESC",
        "cnn_ratio_asc": "f.cnn_area_ratio IS NULL, f.cnn_area_ratio ASC",
    }
    order_by = allowed_sort.get(sort, allowed_sort["frame_idx_asc"])

    count_sql = f"""
        SELECT COUNT(*) FROM frames f
        JOIN segments s ON f.segment_id = s.segment_id
        WHERE {' AND '.join(where)}
    """
    data_sql = f"""
        SELECT f.*, s.`case`, s.video_name, s.video_path, s.start_frame
        FROM frames f
        JOIN segments s ON f.segment_id = s.segment_id
        WHERE {' AND '.join(where)}
        ORDER BY {order_by}
        LIMIT ? OFFSET ?
    """
    query_params = params + [limit, offset]
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute(count_sql, params).fetchone()[0]
        rows = [dict(r) for r in conn.execute(data_sql, query_params).fetchall()]
    for r in rows:
        if r.get("preview_path"):
            r["preview_url"] = f"/api/preview/{r['preview_path']}"
        if r.get("overlay_path"):
            r["overlay_url"] = f"/api/overlay/{r['overlay_path']}"
    return jsonify({"frames": rows, "total": total})


@bp.post("/frames/<segment_id>/<int:frame_idx>/label")
def label_frame(segment_id: str, frame_idx: int):
    data = request.get_json(force=True, silent=True) or {}
    label = data.get("label", "uncertain")
    if label not in FRAME_STATUSES:
        return jsonify({"error": f"invalid label: {label}"}), 400
    update_frame_label(
        segment_id=segment_id,
        frame_idx=frame_idx,
        label=label,
        invalid_reason=data.get("invalid_reason"),
        tags=data.get("tags"),
        notes=data.get("notes"),
        reviewer=data.get("reviewer", "default"),
        session_id=data.get("session_id"),
    )
    return jsonify({"ok": True})


@bp.get("/preview/<path:filepath>")
def serve_preview(filepath: str):
    image_path = _safe_path(PREVIEW_DIR, filepath)
    if image_path is None:
        return jsonify({"error": "invalid path"}), 400
    return _serve_image(image_path)


@bp.get("/overlay/<path:filepath>")
def serve_overlay(filepath: str):
    mask_path = _safe_path(OVERLAY_DIR, filepath)
    if mask_path is None:
        return jsonify({"error": "invalid path"}), 400
    return _serve_mask(mask_path)


@bp.get("/frame_full/<segment_id>/<int:frame_idx>")
def serve_full_frame(segment_id: str, frame_idx: int):
    seg = get_segment(segment_id)
    if not seg:
        return jsonify({"error": "segment not found"}), 404
    case, video_stem = _case_and_stem(segment_id)
    out_dir = SEGMENT_FRAMES_DIR / case / video_stem / segment_id
    out_path = out_dir / f"{case}_{video_stem}_{frame_idx:08d}.jpg"

    # Prefer a fullres_path already recorded in the database.
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT fullres_path FROM frames WHERE segment_id = ? AND frame_idx = ?",
            (segment_id, frame_idx),
        ).fetchone()
    if row and row["fullres_path"]:
        cached = _safe_path(SEGMENT_FRAMES_DIR, row["fullres_path"])
        if cached and cached.exists():
            return send_file(str(cached), mimetype="image/jpeg", max_age=86400)

    if not out_path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(seg["video_path"])
        if not cap.isOpened():
            return jsonify({"error": "cannot open video"}), 500
        # Fully sequential read up to the target frame.  cap.set() on H.264
        # can land on a key frame and either return the wrong image or
        # corrupt subsequent decoding (observed in cases 22 and 23), so we
        # read every frame from the start.  This is slow on first access but
        # guarantees the correct frame.
        target = frame_idx
        current = 0
        frame = None
        while current <= target:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current == target:
                break
            current += 1
        cap.release()
        if frame is None:
            return jsonify({"error": "cannot read frame"}), 500
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        from tools.screening.db import set_frame_paths

        rel_path = str(out_path.relative_to(SEGMENT_FRAMES_DIR))
        try:
            set_frame_paths(segment_id, frame_idx, fullres_path=rel_path)
        except Exception:
            pass
    return send_file(str(out_path), mimetype="image/jpeg", max_age=86400)


@bp.post("/sessions")
def new_session():
    data = request.get_json(force=True, silent=True) or {}
    session_id = start_session(
        action=data.get("action", "segment_review"),
        reviewer=data.get("reviewer", "default"),
        notes=data.get("notes"),
    )
    return jsonify({"session_id": session_id})


@bp.post("/sessions/<session_id>/end")
def finish_session(session_id: str):
    data = request.get_json(force=True, silent=True) or {}
    end_session(
        session_id=session_id,
        segments_reviewed=data.get("segments_reviewed", 0),
        frames_reviewed=data.get("frames_reviewed", 0),
    )
    return jsonify({"ok": True})


@bp.post("/export")
def export():
    data = request.get_json(force=True, silent=True) or {}
    from tools.screening.export_annotations import export_annotations

    path = export_annotations(
        name=data.get("name", "sfy_manual"),
        copy_images=data.get("copy_images", False),
        split_ratio=data.get("split_ratio", 0.8),
        allow_cnn_fallback=data.get("allow_cnn_fallback", False),
    )
    return jsonify({"export_path": str(path)})
