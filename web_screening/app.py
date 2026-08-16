#!/usr/bin/env python3
"""Local Flask web app for SFY video screening and annotation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flask import Flask

from web_screening.routes import api, views


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(views.bp)
    app.register_blueprint(api.bp, url_prefix="/api")

    @app.after_request
    def add_cache_headers(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)
