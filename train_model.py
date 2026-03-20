"""
train_model.py
--------------
Trains the Win Probability model and saves it to disk.
Run this ONCE after generate_data.py.
"""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.ensemble         import RandomForestClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import accuracy_score, classification_report
from sklearn.pipeline         import Pipeline

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURES    = ["runs_left", "balls_left", "wickets_left",
               "current_run_rate", "required_run_rate"]
TARGET      = "result"
DATA_PATH   = "data/ipl_data.csv"
MODEL_PATH  = "models/win_probability_model.pkl"


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV, keep only 2nd-innings rows that have a valid target."""
    df = pd.read_csv(path)

    # Only second-innings rows matter for win probability
    df = df[df["innings"] == 2].copy()

    # Drop any rows where required_run_rate is 0 (corrupted)
    df = df[df["required_run_rate"] > 0]

    # Cap extreme values
    df["required_run_rate"] = df["required_run_rate"].clip(upper=36)
    df["current_run_rate"]  = df["current_run_rate"].clip(upper=36)

    return df


def build_pipeline(model_type: str = "rf") -> Pipeline:
    """Return a sklearn Pipeline with scaler + classifier."""
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators   = 200,
            max_depth      = 8,
            min_samples_leaf = 5,
            random_state   = 42,
            n_jobs         = -1,
        )
    else:
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


def train(data_path: str = DATA_PATH, model_path: str = MODEL_PATH,
          model_type: str = "rf") -> dict:
    """Full train-evaluate-save workflow. Returns metrics dict."""

    print("Loading data …")
    df   = load_and_prepare(data_path)
    X    = df[FEATURES]
    y    = df[TARGET]

    print(f"  Samples : {len(df):,}  |  Class balance : {y.mean():.2%} wins")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training {model_type.upper()} pipeline …")
    pipe = build_pipeline(model_type)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Model saved → {model_path}")

    return {"accuracy": acc, "report": report, "features": FEATURES}


if __name__ == "__main__":
    train()
