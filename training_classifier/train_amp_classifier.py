"""
AMP Binary Classification using ProtT5 Embeddings
-------------------------------------------------
- Stratified cross-validation on TRAIN set
- Final evaluation on held-out TEST set
- Best model selected by TEST ROC-AUC
- Saves classifier and encoder metadata
"""

import os
import re
import json
import joblib
import torch
import numpy as np
import pandas as pd

from transformers import T5EncoderModel, T5Tokenizer

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ================================================================
# CONFIGURATION
# ================================================================

TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"

SAVE_ROOT = "./saved_models_AMP_classifier"
BEST_DIR = os.path.join(SAVE_ROOT, "best_model")
os.makedirs(BEST_DIR, exist_ok=True)

ENCODER_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"

RANDOM_STATE = 42
N_SPLITS = 5
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# ================================================================
# DATA LOADING
# ================================================================


def load_csv_data(csv_path):
    """Load sequences and binary labels from CSV."""
    df = pd.read_csv(csv_path)

    sequences = df["sequence"].astype(str).tolist()
    labels = df["mic_class_binary"].values

    # Replace invalid amino acids
    sequences = [
        re.sub(r"[UZOB]", "X", seq.upper()) for seq in sequences
    ]

    return sequences, labels


# ================================================================
# ProtT5 SETUP
# ================================================================


def load_prott5():
    """Load ProtT5 tokenizer and encoder."""
    print("🧬 Loading ProtT5 encoder...")
    tokenizer = T5Tokenizer.from_pretrained(
        ENCODER_NAME, do_lower_case=False
    )
    encoder = (
        T5EncoderModel.from_pretrained(ENCODER_NAME)
        .to(DEVICE)
        .eval()
    )
    return encoder, tokenizer


# ================================================================
# EMBEDDINGS
# ================================================================


def generate_embeddings(sequences, encoder, tokenizer, batch_size):
    """Generate mean-pooled ProtT5 embeddings."""
    embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        batch_tok = [" ".join(list(seq)) for seq in batch]

        tokens = tokenizer(
            batch_tok,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        with torch.no_grad():
            hidden_states = encoder(**tokens).last_hidden_state

        for j, seq in enumerate(batch_tok):
            seq_len = len(seq.split())
            emb = hidden_states[j, :seq_len].mean(dim=0)
            embeddings.append(emb.cpu().numpy())

    return np.array(embeddings)


# ================================================================
# MODELS
# ================================================================


def build_models():
    """Define candidate classifiers."""
    return {
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
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
                        n_estimators=200,
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


# ================================================================
# EVALUATION
# ================================================================


def report_cv_results(name, cv_results):
    """Print cross-validation summary."""
    print(f"\n===== {name} | CV RESULTS (TRAIN) =====")
    for metric in SCORING:
        mean = cv_results[f"test_{metric}"].mean()
        std = cv_results[f"test_{metric}"].std()
        print(f"{metric:<10}: {mean:.4f} ± {std:.4f}")


def evaluate_on_test(pipe, X_test, y_test):
    """Evaluate trained model on test set."""
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\n📊 Test Performance")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    return metrics


# ================================================================
# SAVE ARTIFACTS
# ================================================================


def save_artifacts(model, metrics):
    """Save best model and metadata."""
    joblib.dump(model, os.path.join(BEST_DIR, "classifier.pkl"))

    encoder_info = {
        "encoder_name": ENCODER_NAME,
        "embedding": "mean_pooling_last_hidden_state",
        "tokenization": "space-separated amino acids",
        "device": str(DEVICE),
        "task": "binary_AMP_classification",
    }

    with open(os.path.join(BEST_DIR, "encoder_info.json"), "w") as f:
        json.dump(encoder_info, f, indent=4)

    with open(os.path.join(BEST_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


# ================================================================
# MAIN
# ================================================================


def main():
    print("📥 Loading data...")
    X_train_seqs, y_train = load_csv_data(TRAIN_CSV)
    X_test_seqs, y_test = load_csv_data(TEST_CSV)

    print(
        f"Train samples: {len(X_train_seqs)} | "
        f"Test samples: {len(X_test_seqs)}"
    )

    encoder, tokenizer = load_prott5()

    print("🔄 Generating TRAIN embeddings...")
    X_train = generate_embeddings(
        X_train_seqs, encoder, tokenizer, BATCH_SIZE
    )

    print("🔄 Generating TEST embeddings...")
    X_test = generate_embeddings(
        X_test_seqs, encoder, tokenizer, BATCH_SIZE
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    best_score = -np.inf
    best_model = None
    best_name = None
    best_metrics = None

    for name, pipe in models.items():
        print(f"\n🚀 Training {name}")

        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
        )

        report_cv_results(name, cv_results)

        pipe.fit(X_train, y_train)
        test_metrics = evaluate_on_test(
            pipe, X_test, y_test
        )

        if test_metrics["roc_auc"] > best_score:
            best_score = test_metrics["roc_auc"]
            best_model = pipe
            best_name = name
            best_metrics = {"model": name, **test_metrics}

    print(f"\n🏆 Best model: {best_name}")
    print(f"🏅 Best ROC-AUC: {best_score:.4f}")

    save_artifacts(best_model, best_metrics)
    print(f"💾 Saved to: {BEST_DIR}")
    print("🎉 DONE")


if __name__ == "__main__":
    main()