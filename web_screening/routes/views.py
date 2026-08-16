"""HTML page routes."""

from flask import Blueprint, render_template

from tools.screening.db import get_segment, get_segments, get_summary

bp = Blueprint("views", __name__)


@bp.get("/")
def dashboard():
    summary = get_summary()
    return render_template("dashboard.html", summary=summary)


@bp.get("/segments")
def segment_reviewer():
    return render_template("segment_reviewer.html")


@bp.get("/frames_old/<segment_id>")
def frame_reviewer(segment_id: str):
    segment = get_segment(segment_id)
    return render_template("frame_reviewer.html", segment=segment)


@bp.get("/frames")
def frame_browser():
    return render_template("frame_browser.html")
