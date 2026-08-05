import os
import sqlite3
import random
import string
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, jsonify, g
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from detection_engine import analyze_leaf
from fertilizer_data import get_treatment, CROPS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "crop_deficiency.db")
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

# Admin credentials - env-overridable for safe deployment
ADMIN_USER = os.environ.get("ADMIN_USER", "host")
ADMIN_PASS_HASH_ENV = os.environ.get("ADMIN_PASS_HASH")  # optional pre-hashed


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'admin',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            crop_type TEXT NOT NULL,
            image_filename TEXT,
            deficiency_type TEXT,
            confidence REAL,
            severity_level TEXT,
            affected_area_pct REAL,
            green_pct REAL,
            yellow_pct REAL,
            brown_pct REAL,
            purple_pct REAL,
            visual_symptoms TEXT,
            immediate_action TEXT,
            recommended_fertilizer TEXT,
            application_method TEXT,
            dosage TEXT,
            recovery_time TEXT,
            risk_level TEXT,
            overall_health TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fertilizer_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            crop_type TEXT NOT NULL,
            deficiency_type TEXT NOT NULL,
            recommended_fertilizer TEXT,
            application_method TEXT,
            dosage TEXT,
            recovery_time TEXT,
            risk_level TEXT,
            UNIQUE(crop_type, deficiency_type)
        )
    """)
    conn.commit()

    # Seed admin user if not present
    cur.execute("SELECT id FROM users WHERE username = ?", (ADMIN_USER,))
    if cur.fetchone() is None:
        pw_hash = ADMIN_PASS_HASH_ENV or generate_password_hash(
            os.environ.get("ADMIN_PASS", "CropGuard@2026")
        )
        cur.execute(
            "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
            (ADMIN_USER, "admin@cropguard.local", pw_hash, "admin"),
        )
        conn.commit()

    # Seed fertilizer_rules table from fertilizer_data.py if empty
    cur.execute("SELECT COUNT(*) FROM fertilizer_rules")
    if cur.fetchone()[0] == 0:
        from fertilizer_data import FERTILIZER_RULES
        for (crop, deficiency), plan in FERTILIZER_RULES.items():
            cur.execute("""
                INSERT OR IGNORE INTO fertilizer_rules
                (crop_type, deficiency_type, recommended_fertilizer, application_method,
                 dosage, recovery_time, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                crop, deficiency, plan["recommended_fertilizer"],
                plan["application_method"], plan["dosage"],
                plan["recovery_time"], plan["risk_level"]
            ))
        conn.commit()

    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def generate_report_id():
    suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    return f"CG-{suffix}"


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", crops=CROPS)


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html", crops=CROPS)

    crop_type = request.form.get("crop_type", "").strip()
    file = request.files.get("leaf_image")

    if not crop_type:
        flash("Please select a crop type.", "danger")
        return redirect(url_for("detect"))
    if not file or file.filename == "":
        flash("Please upload a leaf image.", "danger")
        return redirect(url_for("detect"))
    if not allowed_file(file.filename):
        flash("Unsupported file type. Use PNG, JPG, JPEG, or WEBP.", "danger")
        return redirect(url_for("detect"))

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    unique_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(filepath)

    try:
        result = analyze_leaf(filepath, crop_type)
    except Exception as e:
        flash(f"Could not analyze image: {e}", "danger")
        return redirect(url_for("detect"))

    deficiency = result["deficiency_type"]
    overall_health = "Healthy" if deficiency == "Healthy" else "Deficient"

    if deficiency == "Healthy":
        treatment = dict(
            visual_symptoms="No visible signs of nutrient stress detected.",
            immediate_action="Continue regular fertilization and monitoring schedule.",
            recommended_fertilizer="Maintain balanced NPK program",
            application_method="As per standard crop calendar",
            dosage="Standard maintenance dose",
            recovery_time="N/A",
            risk_level="None",
        )
    else:
        treatment = get_treatment(crop_type, deficiency)

    report_id = generate_report_id()
    db = get_db()
    db.execute("""
        INSERT INTO predictions (
            report_id, crop_type, image_filename, deficiency_type, confidence,
            severity_level, affected_area_pct, green_pct, yellow_pct, brown_pct,
            purple_pct, visual_symptoms, immediate_action, recommended_fertilizer,
            application_method, dosage, recovery_time, risk_level, overall_health
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report_id, crop_type, unique_name, deficiency, result["confidence"],
        result["severity_level"], result["affected_area_pct"], result["green_pct"],
        result["yellow_pct"], result["brown_pct"], result["purple_pct"],
        treatment["visual_symptoms"], treatment["immediate_action"],
        treatment["recommended_fertilizer"], treatment["application_method"],
        treatment["dosage"], treatment["recovery_time"], treatment["risk_level"],
        overall_health,
    ))
    db.commit()

    return redirect(url_for("results", report_id=report_id))


@app.route("/results/<report_id>")
def results(report_id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM predictions WHERE report_id = ?", (report_id,)
    ).fetchone()
    if row is None:
        flash("Report not found.", "danger")
        return redirect(url_for("detect"))
    return render_template("results.html", r=row)


@app.route("/reports")
def reports():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    return render_template("reports.html", reports=rows)


@app.route("/dashboard")
def dashboard():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    healthy = db.execute(
        "SELECT COUNT(*) c FROM predictions WHERE overall_health='Healthy'"
    ).fetchone()["c"]
    deficient = total - healthy
    return render_template(
        "dashboard.html", total=total, healthy=healthy, deficient=deficient
    )


@app.route("/results/<report_id>/pdf")
def download_pdf(report_id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM predictions WHERE report_id = ?", (report_id,)
    ).fetchone()
    if row is None:
        flash("Report not found.", "danger")
        return redirect(url_for("detect"))

    pdf_path = os.path.join(UPLOAD_FOLDER, f"{report_id}.pdf")
    build_pdf(row, pdf_path)
    return send_file(pdf_path, as_attachment=True, download_name=f"{report_id}_CropGuard_Report.pdf")


def build_pdf(row, path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#1b5e20")
    )
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=20 * mm)
    elements = []

    elements.append(Paragraph("CropGuard AI — Diagnostic Certificate", title_style))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Report ID: <b>{row['report_id']}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Date: {row['created_at']}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    data = [
        ["Crop Type", row["crop_type"].title()],
        ["Diagnosis", row["deficiency_type"]],
        ["Confidence", f"{row['confidence']}%"],
        ["Severity Level", row["severity_level"]],
        ["Affected Leaf Area", f"{row['affected_area_pct']}%"],
        ["Overall Health", row["overall_health"]],
    ]
    t = Table(data, colWidths=[160, 300])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Spectral Analysis (HSV)", styles["Heading3"]))
    spec_data = [
        ["Green (Chlorophyll)", f"{row['green_pct']}%"],
        ["Yellow (Chlorosis)", f"{row['yellow_pct']}%"],
        ["Brown (Necrosis)", f"{row['brown_pct']}%"],
        ["Purple (Anthocyanin)", f"{row['purple_pct']}%"],
    ]
    t2 = Table(spec_data, colWidths=[160, 300])
    t2.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Treatment & Fertilizer Action Plan", styles["Heading3"]))
    plan_data = [
        ["Visual Symptoms", row["visual_symptoms"] or "-"],
        ["Immediate Action", row["immediate_action"] or "-"],
        ["Recommended Fertilizer", row["recommended_fertilizer"] or "-"],
        ["Application Method", row["application_method"] or "-"],
        ["Dosage", row["dosage"] or "-"],
        ["Recovery Time", row["recovery_time"] or "-"],
        ["Risk Level", row["risk_level"] or "-"],
    ]
    t3 = Table(plan_data, colWidths=[160, 300])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#fff8e1")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t3)
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(
        "This is an AI/OpenCV-assisted diagnostic aid, not a substitute for "
        "professional agronomic advice. Consult a certified agronomist for "
        "critical treatment decisions.", styles["Italic"]
    ))

    doc.build(elements)


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html")

    username = request.form.get("username", "")
    password = request.form.get("password", "")
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

    if user and check_password_hash(user["password"], password):
        session["admin_logged_in"] = True
        session["admin_username"] = username
        return redirect(url_for("admin_dashboard"))

    flash("Invalid username or password.", "danger")
    return redirect(url_for("admin_login"))


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
@login_required
def admin_dashboard():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM predictions").fetchone()["c"]
    healthy = db.execute(
        "SELECT COUNT(*) c FROM predictions WHERE overall_health='Healthy'"
    ).fetchone()["c"]
    deficient = total - healthy

    since = (datetime.now() - timedelta(days=1)).isoformat()
    daily = db.execute(
        "SELECT COUNT(*) c FROM predictions WHERE created_at >= ?", (since,)
    ).fetchone()["c"]

    top_deficiency_row = db.execute("""
        SELECT deficiency_type, COUNT(*) c FROM predictions
        WHERE overall_health = 'Deficient'
        GROUP BY deficiency_type ORDER BY c DESC LIMIT 1
    """).fetchone()
    top_deficiency = top_deficiency_row["deficiency_type"] if top_deficiency_row else "N/A"

    top_crop_row = db.execute("""
        SELECT crop_type, COUNT(*) c FROM predictions
        GROUP BY crop_type ORDER BY c DESC LIMIT 1
    """).fetchone()
    top_crop = top_crop_row["crop_type"].title() if top_crop_row else "N/A"

    deficiency_dist = db.execute("""
        SELECT deficiency_type, COUNT(*) c FROM predictions
        GROUP BY deficiency_type ORDER BY c DESC
    """).fetchall()

    crop_dist = db.execute("""
        SELECT crop_type, COUNT(*) c FROM predictions
        GROUP BY crop_type ORDER BY c DESC
    """).fetchall()

    rows = db.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 100"
    ).fetchall()

    return render_template(
        "admin_dashboard.html",
        total=total, healthy=healthy, deficient=deficient, daily=daily,
        top_deficiency=top_deficiency, top_crop=top_crop,
        deficiency_dist=deficiency_dist, crop_dist=crop_dist, rows=rows,
    )


@app.route("/admin/fertilizers", methods=["GET", "POST"])
@login_required
def admin_fertilizers():
    db = get_db()
    if request.method == "POST":
        rule_id = request.form.get("rule_id")
        db.execute("""
            UPDATE fertilizer_rules SET
                recommended_fertilizer = ?, application_method = ?,
                dosage = ?, recovery_time = ?, risk_level = ?
            WHERE id = ?
        """, (
            request.form.get("recommended_fertilizer"),
            request.form.get("application_method"),
            request.form.get("dosage"),
            request.form.get("recovery_time"),
            request.form.get("risk_level"),
            rule_id,
        ))
        db.commit()
        flash("Fertilizer rule updated.", "success")
        return redirect(url_for("admin_fertilizers"))

    rules = db.execute(
        "SELECT * FROM fertilizer_rules ORDER BY crop_type, deficiency_type"
    ).fetchall()
    return render_template("admin_fertilizers.html", rules=rules)


@app.route("/admin/password", methods=["GET", "POST"])
@login_required
def admin_password():
    if request.method == "POST":
        current = request.form.get("current_password", "")
        new = request.form.get("new_password", "")
        confirm = request.form.get("confirm_password", "")
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (session["admin_username"],)
        ).fetchone()

        if not check_password_hash(user["password"], current):
            flash("Current password is incorrect.", "danger")
        elif new != confirm:
            flash("New passwords do not match.", "danger")
        elif len(new) < 8:
            flash("New password must be at least 8 characters.", "danger")
        else:
            db.execute(
                "UPDATE users SET password = ? WHERE username = ?",
                (generate_password_hash(new), session["admin_username"]),
            )
            db.commit()
            flash("Password updated successfully.", "success")
        return redirect(url_for("admin_password"))

    return render_template("admin_password.html")


if __name__ == "__main__":
    init_db()
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
else:
    # Ensures DB is initialized when run via gunicorn too
    init_db()
