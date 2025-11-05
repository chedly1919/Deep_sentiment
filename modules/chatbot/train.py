import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump
from pathlib import Path
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

DATA_PATH = Path("data/training.1600000.processed.noemoticon.csv")
MODEL_PATH = Path("models/sentiment.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_data(n_rows=100000):  # you can adjust number of rows (start small!)
    headers = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(DATA_PATH, encoding="latin-1", names=headers)
    df = df[["text", "target"]]
    df["label"] = df["target"].map({0: "neg", 4: "pos"})
    df.dropna(subset=["text"], inplace=True)
    df = df.sample(n=n_rows, random_state=42)
    return df[["text", "label"]]

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1,2),
            max_features=50000
        )),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    dump(pipe, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()
