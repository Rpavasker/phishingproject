from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sqlite3
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
DB_PATH = BASE_DIR / "phishing_app.db"

app = FastAPI(title="Phishing Detection Website")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            input_text TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model.pkl not found at: {MODEL_PATH}")

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"vectorizer.pkl not found at: {VECTORIZER_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


init_db()
model, vectorizer = load_model()


@app.get("/", response_class=HTMLResponse)
def home(request: Request, error: str = "", message: str = ""):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "error": error,
            "message": message
        }
    )


@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, message: str = ""):
    return templates.TemplateResponse(
        "signup.html",
        {
            "request": request,
            "message": message
        }
    )


@app.post("/signup", response_class=HTMLResponse)
def signup(request: Request, username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    password = password.strip()

    if not username or not password:
        return templates.TemplateResponse(
            "signup.html",
            {
                "request": request,
                "message": "Username and password are required."
            }
        )

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()

        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "message": "Signup successful. Please login.",
                "error": ""
            }
        )

    except sqlite3.IntegrityError:
        return templates.TemplateResponse(
            "signup.html",
            {
                "request": request,
                "message": "Username already exists. Try another one."
            }
        )

    except Exception as e:
        return HTMLResponse(
            content=f"Signup failed: {str(e)}",
            status_code=500
        )

    finally:
        if conn:
            conn.close()


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    password = password.strip()

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    user = cur.fetchone()
    conn.close()

    if user:
        return RedirectResponse(url=f"/dashboard?username={username}", status_code=303)

    return RedirectResponse(url="/?error=Invalid username or password", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    username: str = "",
    result: str = "",
    confidence: str = "",
    text: str = ""
):
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "result": result,
            "confidence": confidence,
            "text": text
        }
    )


@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    username: str = Form(...),
    text: str = Form(...)
):
    cleaned_text = text.strip()

    if len(cleaned_text) < 5:
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "username": username,
                "result": "Please enter a longer message or URL.",
                "confidence": "",
                "text": cleaned_text
            }
        )

    transformed = vectorizer.transform([cleaned_text])
    pred = int(model.predict(transformed)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(transformed)[0]
        confidence_value = float(max(probs)) * 100
    else:
        confidence_value = 0.0

    result = "PHISHING ⚠️" if pred == 1 else "LEGIT ✅"

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (username, input_text, prediction, confidence) VALUES (?, ?, ?, ?)",
        (username, cleaned_text, result, confidence_value)
    )
    conn.commit()
    conn.close()

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "result": result,
            "confidence": f"{confidence_value:.2f}%",
            "text": cleaned_text
        }
    )


@app.get("/history", response_class=HTMLResponse)
def history(request: Request, username: str = ""):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM predictions WHERE username = ? ORDER BY id DESC",
        (username,)
    )
    rows = cur.fetchall()
    conn.close()

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "username": username,
            "rows": rows
        }
    )


@app.get("/health")
def health():
    return {"status": "running"}