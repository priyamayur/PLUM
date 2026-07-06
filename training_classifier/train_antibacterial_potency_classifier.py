# ============================================================
# AMP MIC Classification (Multiclass)
# - ProtT5 embeddings
# - Multiple models
# - Select & save BEST model based on CV F1
# ============================================================

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
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# ============================================================
# PATHS & SETTINGS
# ============================================================

# DATA_PATH = "./data/generic_all_mic_data.csv"
#DATA_PATH = "./data/binary_amp_dataset_april_2026.csv"
DATA_PATH = "./data/MIC_gram_neutral_amp_dataset_may_2026_2.csv"
#DATA_PATH = "./data/MIC_gram-positive_amp_dataset_may_2026.csv"
print(f"📊 Loading data from: {DATA_PATH}")
# SAVE_ROOT = "./saved_models_MIC"
SAVE_ROOT = "./saved_models_neutral"
# SAVE_ROOT = "./saved_models_GramPos"
BEST_DIR = os.path.join(SAVE_ROOT, "best_model")
os.makedirs(BEST_DIR, exist_ok=True)

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["mic_class", "sequence"])

sequences = df["sequence"].values
labels = df["mic_class"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print("Classes:", label_encoder.classes_)

# ============================================================
# 2️⃣ LOAD ProtT5
# ============================================================

print("🧬 Loading ProtT5...")
tokenizer = T5Tokenizer.from_pretrained(ENCODER_NAME, do_lower_case=False)
encoder = T5EncoderModel.from_pretrained(ENCODER_NAME).to(DEVICE).eval()

def get_embedding(sequence):
    """Compute mean pooled ProtT5 embedding for a protein sequence"""
    seq = re.sub(r"[UZOB]", "X", sequence.upper())
    seq = " ".join(list(seq))
    tokens = tokenizer(seq, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = encoder(**tokens).last_hidden_state.squeeze(0)
    return out.mean(dim=0).cpu().numpy()

# ============================================================
# 3️⃣ COMPUTE EMBEDDINGS
# ============================================================

print("🔄 Computing embeddings...")
X_list = []
start = time.time()

for i, seq in enumerate(sequences, 1):
    try:
        X_list.append(get_embedding(seq))
    except Exception as e:
        print(f"Skipping seq {i}: {e}")
    if i % 50 == 0 or i == len(sequences):
        print(f"Processed {i}/{len(sequences)}")

X = np.array(X_list)
print(f"Finished embeddings in {(time.time() - start)/60:.2f} min")
print("Embedding shape:", X.shape)

# ============================================================
# 4️⃣ CLASS-BALANCED TRAIN / TEST SPLIT (80/20)
# ============================================================

np.random.seed(RANDOM_STATE)
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

for cls in np.unique(y):
    idx = np.where(y == cls)[0]
    np.random.shuffle(idx)
    n_test = max(1, int(0.2 * len(idx)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_train_list.append(X[train_idx])
    y_train_list.append(y[train_idx])
    X_test_list.append(X[test_idx])
    y_test_list.append(y[test_idx])

X_train = np.vstack(X_train_list)
y_train = np.hstack(y_train_list)
X_test = np.vstack(X_test_list)
y_test = np.hstack(y_test_list)

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)
print("Train class distribution:", np.bincount(y_train))
print("Test class distribution :", np.bincount(y_test))

# ============================================================
# 5️⃣ DEFINE MODELS
# ============================================================
RANDOM_STAT = 42
models = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STAT, class_weight="balanced"))
    ]),
    "Random_Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"))
    ]),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000, random_state=RANDOM_STATE))
    ])
}

# ============================================================
# 6️⃣ TRAIN, CROSS-VALIDATE, AND SELECT BEST MODEL
# ============================================================

best_score = -np.inf
best_model_name = None
best_pipe = None
best_metrics = None

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring_metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
# scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for name, pipe in models.items():
    print(f"\n🚀 Training {name}")

    # -------------------------
    # 🔁 CROSS-VALIDATION
    # -------------------------
    cv_results = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=False
    )

    print("CV Accuracy   : {:.4f} ± {:.4f}".format(cv_results["test_accuracy"].mean(), cv_results["test_accuracy"].std()))
    print("CV Precision  : {:.4f} ± {:.4f}".format(cv_results["test_precision_weighted"].mean(), cv_results["test_precision_weighted"].std()))
    print("CV Recall     : {:.4f} ± {:.4f}".format(cv_results["test_recall_weighted"].mean(), cv_results["test_recall_weighted"].std()))
    print("CV F1         : {:.4f} ± {:.4f}".format(cv_results["test_f1_weighted"].mean(), cv_results["test_f1_weighted"].std()))

    # -------------------------
    # 🧠 TRAIN FINAL MODEL
    # -------------------------
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except:
        roc_auc = np.nan

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # -------------------------
    # 🏆 MODEL SELECTION BASED ON CV F1
    # -------------------------
    cv_f1_mean = cv_results["test_f1_weighted"].mean()
    cv_f1_std = cv_results["test_f1_weighted"].std()
    score_for_selection = cv_f1_mean

    if score_for_selection > best_score:
        best_score = score_for_selection
        best_model_name = name
        best_pipe = pipe
        best_metrics = {
            "model": name,
            "cv_f1_mean": cv_f1_mean,
            "cv_f1_std": cv_f1_std,
            "cv_accuracy_mean": cv_results["test_accuracy"].mean(),
            "cv_accuracy_std": cv_results["test_accuracy"].std(),
            "cv_precision_mean": cv_results["test_precision_weighted"].mean(),
            "cv_precision_std": cv_results["test_precision_weighted"].std(),
            "cv_recall_mean": cv_results["test_recall_weighted"].mean(),
            "cv_recall_std": cv_results["test_recall_weighted"].std(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        }

# ============================================================
# 7️⃣ SAVE BEST MODEL + METADATA
# ============================================================

print(f"\n🏆 Best model: {best_model_name}")
print(f"🏅 Best score (CV F1): {best_score:.4f}")

# Save pipeline (includes scaler)
joblib.dump(best_pipe, os.path.join(BEST_DIR, "classifier.pkl"))

# Save label encoder (needed for inference)
joblib.dump(label_encoder, os.path.join(BEST_DIR, "label_encoder.pkl"))

# Save encoder info
encoder_info = {
    "encoder_name": ENCODER_NAME,
    "embedding": "mean_pooling_last_hidden_state",
    "task": "multiclass_MIC"
}
with open(os.path.join(BEST_DIR, "encoder_info.json"), "w") as f:
    json.dump(encoder_info, f, indent=4)

# Save metrics
with open(os.path.join(BEST_DIR, "metrics.json"), "w") as f:
    json.dump(best_metrics, f, indent=4)

print(f"\n💾 Saved best model to: {BEST_DIR}")
print("🎉 DONE")