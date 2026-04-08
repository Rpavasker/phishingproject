import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "spam.csv"
MODEL_FILE = BASE_DIR / "model.pkl"
VECTORIZER_FILE = BASE_DIR / "vectorizer.pkl"


def find_columns(df):
    # common SMS spam dataset format
    if "v1" in df.columns and "v2" in df.columns:
        return "v2", "v1"

    cols = [c.lower().strip() for c in df.columns]

    text_candidates = ["text", "email", "message", "body", "content"]
    label_candidates = ["label", "target", "class", "category"]

    text_col = None
    label_col = None

    for original, lower in zip(df.columns, cols):
        if lower in text_candidates and text_col is None:
            text_col = original
        if lower in label_candidates and label_col is None:
            label_col = original

    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Found columns: {list(df.columns)}")

    return text_col, label_col


def map_label(value):
    s = str(value).strip().lower()

    if s in ["spam", "phishing", "1", "true", "yes"]:
        return 1   # phishing/suspicious
    return 0       # legit/safe


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, encoding="latin-1")

    text_col, label_col = find_columns(df)

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].apply(map_label)

    df = df[df[text_col].str.len() > 3]

    print("Using text column:", text_col)
    print("Using label column:", label_col)
    print("\nLabel counts:")
    print(df[label_col].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved model to: {MODEL_FILE}")
    print(f"Saved vectorizer to: {VECTORIZER_FILE}")


if __name__ == "__main__":
    main()