"""
AMP MIC Classification (Multiclass)
----------------------------------
- ProtT5 embeddings
- Multiple ML models
- Automatic best-model selection
- Saves model, label encoder, and metadata

Author: <Your Name>
"""

import os
import re
import json
import time
import joblib
import numpy as np
import pandas as pd
import torch

from transformers import T5EncoderModel, T5Tokenizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "./data/generic_all_mic_data.csv"

SAVE_ROOT = "./saved_models_MIC"
BEST_DIR = os.path.join(SAVE_ROOT, "best_model")
os.makedirs(BEST_DIR, exist_ok=True)

RANDOM_STATE = 42
ENCODER_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATA LOADING
# ============================================================


def load_data(path):
    """Load dataset and encode labels."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["mic_class", "sequence"])

    sequences = df["sequence"].values
    labels = df["mic_class"].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    print("Classes:", label_encoder.classes_)
    return sequences, y, label_encoder


# ============================================================
# ProtT5 EMBEDDINGS
# ============================================================


def load_prott5():
    """Load ProtT5 tokenizer and encoder."""
    print("🧬 Loading ProtT5...")
    tokenizer = T5Tokenizer.from_pretrained(
        ENCODER_NAME, do_lower_case=False
    )
    encoder = (
        T5EncoderModel.from_pretrained(ENCODER_NAME)
        .to(DEVICE)
        .eval()
    )
    return tokenizer, encoder


def get_embedding(sequence, tokenizer, encoder):
    """Compute mean-pooled ProtT5 embedding for a sequence."""
    seq = re.sub(r"[UZOB]", "X", sequence.upper())
    seq = " ".join(list(seq))

    tokens = tokenizer(seq, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        hidden_states = encoder(**tokens).last_hidden_state.squeeze(0)

    return hidden_states.mean(dim=0).cpu().numpy()


def compute_embeddings(sequences, tokenizer, encoder):
    """Compute embeddings for all sequences."""
    print("🔄 Computing embeddings...")
    start = time.time()

    embeddings = []
    for i, seq in enumerate(sequences, 1):
        try:
            embeddings.append(get_embedding(seq, tokenizer, encoder))
        except Exception as e:
            print(f"⚠️ Skipping sequence {i}: {e}")

        if i % 50 == 0 or i == len(sequences):
            print(f"Processed {i}/{len(sequences)}")

    X = np.array(embeddings)
    print(
        f"Finished embeddings in {(time.time() - start) / 60:.2f} min"
    )
    print("Embedding shape:", X.shape)
    return X


# ============================================================
# TRAIN / TEST SPLIT (CLASS-BALANCED)
# ============================================================


def balanced_train_test_split(X, y, test_ratio=0.2):
    """Create class-balanced train/test split."""
    np.random.seed(RANDOM_STATE)

    X_train, X_test, y_train, y_test = [], [], [], []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)

        n_test = max(1, int(test_ratio * len(idx)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_train.append(X[train_idx])
        y_train.append(y[train_idx])
        X_test.append(X[test_idx])
        y_test.append(y[test_idx])

    return (
        np.vstack(X_train),
        np.hstack(y_train),
        np.vstack(X_test),
        np.hstack(y_test),
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================


def build_models():
    """Define ML pipelines."""
    return {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=10,
                        gamma="scale",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random_Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 256),
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# TRAINING & EVALUATION
# ============================================================


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Train models, evaluate, and select the best."""
    best_score = -np.inf
    best_model = None
    best_name = None
    best_metrics = None

    for name, pipe in models.items():
        print(f"\n🚀 Training {name}")
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "f1": f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
        }

        try:
            metrics["roc_auc"] = roc_auc_score(
                y_test,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            metrics["roc_auc"] = np.nan

        for k, v in metrics.items():
            print(f"{k.upper():10s}: {v:.4f}")

        score = (
            metrics["roc_auc"]
            if not np.isnan(metrics["roc_auc"])
            else metrics["f1"]
        )

        if score > best_score:
            best_score = score
            best_model = pipe
            best_name = name
            best_metrics = {"model": name, **metrics}

    return best_name, best_model, best_score, best_metrics


# ============================================================
# SAVE ARTIFACTS
# ============================================================


def save_artifacts(model, label_encoder, metrics):
    """Save trained model and metadata."""
    joblib.dump(model, os.path.join(BEST_DIR, "classifier.pkl"))
    joblib.dump(
        label_encoder, os.path.join(BEST_DIR, "label_encoder.pkl")
    )

    with open(os.path.join(BEST_DIR, "encoder_info.json"), "w") as f:
        json.dump(
            {
                "encoder_name": ENCODER_NAME,
                "embedding": "mean_pooling_last_hidden_state",
                "task": "multiclass_MIC",
            },
            f,
            indent=4,
        )

    with open(os.path.join(BEST_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


# ============================================================
# MAIN
# ============================================================


def main():
    sequences, y, label_encoder = load_data(DATA_PATH)

    tokenizer, encoder = load_prott5()
    X = compute_embeddings(sequences, tokenizer, encoder)

    X_train, y_train, X_test, y_test = balanced_train_test_split(X, y)

    print("Train size:", X_train.shape)
    print("Test size :", X_test.shape)

    models = build_models()
    best_name, best_model, best_score, best_metrics = evaluate_models(
        models, X_train, y_train, X_test, y_test
    )

    print(f"\n🏆 Best model: {best_name}")
    print(f"🏅 Best score: {best_score:.4f}")

    save_artifacts(best_model, label_encoder, best_metrics)
    print(f"💾 Saved to: {BEST_DIR}")
    print("🎉 DONE")


if __name__ == "__main__":
    main()
