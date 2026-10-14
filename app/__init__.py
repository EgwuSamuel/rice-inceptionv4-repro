"""Flask web app: upload a rice leaf photo, get a diagnosis and management advice.

Run:  RICEDX_MODEL_DIR=results/app_bundle python -m app
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

HERE = Path(__file__).resolve().parent
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id),
  image_file TEXT NOT NULL,
  label TEXT NOT NULL,
  confidence REAL NOT NULL,
  abstained INTEGER NOT NULL,
  created_at TEXT NOT NULL
);
"""


def create_app(classifier=None, config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.update(
        SECRET_KEY=os.environ.get("RICEDX_SECRET_KEY") or secrets.token_hex(32),
        DATABASE=os.environ.get("RICEDX_DB", str(Path(app.instance_path) / "ricedx.sqlite")),
        MODEL_DIR=os.environ.get("RICEDX_MODEL_DIR", str(HERE.parent / "results" / "app_bundle")),
        CONFIDENCE_THRESHOLD=float(os.environ.get("RICEDX_THRESHOLD", "0.70")),
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        UPLOAD_DIR=str(HERE / "static" / "uploads"),
    )
    if config:
        app.config.update(config)
    Path(app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    knowledge = json.loads((HERE / "knowledge.json").read_text(encoding="utf-8"))

    state = {"classifier": classifier, "error": None}

    def get_classifier():
        if state["classifier"] is None and state["error"] is None:
            try:
                from .inference import OnnxClassifier

                state["classifier"] = OnnxClassifier(app.config["MODEL_DIR"])
            except Exception as exc:  # model bundle missing or unreadable
                state["error"] = f"Model could not be loaded from {app.config['MODEL_DIR']}: {exc}"
        return state["classifier"]

    # ---------------- database ---------------- #
    def db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DATABASE"])
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    with app.app_context():
        db().executescript(SCHEMA)
        db().commit()

    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                return redirect(url_for("login"))
            return view(*args, **kwargs)

        return wrapped

    # ---------------- auth ---------------- #
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if len(username) < 3 or len(password) < 8:
                flash("Username needs 3+ characters and password 8+ characters.", "danger")
                return render_template("register.html"), 400
            try:
                db().execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db().commit()
            except sqlite3.IntegrityError:
                flash("Username already exists.", "danger")
                return render_template("register.html"), 409
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            row = db().execute(
                "SELECT id, password_hash FROM users WHERE username = ?",
                (request.form.get("username", "").strip(),),
            ).fetchone()
            if row and check_password_hash(row["password_hash"], request.form.get("password", "")):
                session.clear()
                session["user_id"] = row["id"]
                session["username"] = request.form["username"].strip()
                return redirect(url_for("index"))
            flash("Invalid username or password.", "danger")
            return render_template("login.html"), 401
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ---------------- prediction ---------------- #
    @app.route("/")
    def home():
        return redirect(url_for("index" if "user_id" in session else "login"))

    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def index():
        result = None
        if request.method == "POST":
            file = request.files.get("file")
            if file is None or file.filename == "":
                flash("Choose an image first.", "warning")
                return render_template("index.html", result=None), 400
            ext = Path(secure_filename(file.filename)).suffix.lower()
            if ext not in ALLOWED_EXTS:
                flash("Unsupported file type.", "danger")
                return render_template("index.html", result=None), 400
            try:
                img = Image.open(file.stream)
                img.load()
            except (UnidentifiedImageError, OSError):
                flash("The file is not a readable image.", "danger")
                return render_template("index.html", result=None), 400
            clf = get_classifier()
            if clf is None:
                flash(state["error"], "danger")
                return render_template("index.html", result=None), 503
            name = f"{uuid.uuid4().hex}{ext}"
            img.convert("RGB").save(Path(app.config["UPLOAD_DIR"]) / name)
            pred = clf.predict(img)
            abstained = pred["confidence"] < app.config["CONFIDENCE_THRESHOLD"]
            db().execute(
                "INSERT INTO predictions (user_id, image_file, label, confidence, abstained, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session["user_id"], name, pred["label"], pred["confidence"], int(abstained),
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            db().commit()
            result = {
                **pred,
                "abstained": abstained,
                "image_url": url_for("static", filename=f"uploads/{name}"),
                "advice": None if abstained else knowledge.get(pred["label"]),
                "threshold": app.config["CONFIDENCE_THRESHOLD"],
            }
        return render_template("index.html", result=result)

    @app.route("/history")
    @login_required
    def history():
        rows = db().execute(
            "SELECT image_file, label, confidence, abstained, created_at FROM predictions "
            "WHERE user_id = ? ORDER BY id DESC LIMIT 50",
            (session["user_id"],),
        ).fetchall()
        return render_template("history.html", rows=rows)

    return app
