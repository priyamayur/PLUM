from Bio import SeqIO
import joblib
import torch
import numpy as np
import json
from transformers import T5EncoderModel, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# -------------------- Settings --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEST_MODEL_DIR = "./saved_models_AMP_classifier/best_model"
CLASSIFIER_PATH = f"{BEST_MODEL_DIR}/classifier.pkl"
ENCODER_INFO_PATH = f"{BEST_MODEL_DIR}/encoder_info.json"

AMP_THRESHOLD = 0.5
NONAMP_THRESHOLDS = [0.5, 0.7, 0.8]

fasta_files = {
    "HydrAMP": "./data/HydrAMP_generated_nonAMP_45k.fasta",
    "PLUM": "./data/PLUM_generated_nonAMP_func0_45k.fasta",
    "Baseline1": "./data/Baseline1_generated_AMP_func0_45k.fasta",
    "Baseline2": "./data/Baseline2_generated_AMP_func0_45k.fasta"
}

# -------------------- Load models ONCE --------------------
classifier = joblib.load(CLASSIFIER_PATH)

with open(ENCODER_INFO_PATH) as f:
    encoder_name = json.load(f)["encoder_name"]

encoder = T5EncoderModel.from_pretrained(encoder_name).to(DEVICE).eval()
tokenizer = T5Tokenizer.from_pretrained(encoder_name, do_lower_case=False)

# -------------------- Helper functions --------------------
def generate_embeddings(sequences, batch_size=64):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = [" ".join(seq) for seq in sequences[i:i + batch_size]]
        tokens = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            out = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1)

        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def embedding_diversity(X):
    if len(X) < 2:
        return np.nan
    sim = cosine_similarity(X)
    iu = np.triu_indices_from(sim, k=1)
    return float(1.0 - sim[iu].mean())


# -------------------- Embed & cache ONCE --------------------
cache = {}

for name, fasta in fasta_files.items():
    records = list(SeqIO.parse(fasta, "fasta"))
    sequences = [str(r.seq).upper() for r in records]

    if not sequences:
        continue

    print(f"⏳ Embedding {name} ({len(sequences)} peptides)")
    X = generate_embeddings(sequences)
    lengths = np.array([len(s) for s in sequences])

    amp_probs = classifier.predict_proba(X)[:, 1]
    nonamp_probs = 1.0 - amp_probs
    nonamp_mask = nonamp_probs >= AMP_THRESHOLD

    cache[name] = {
        "X": X,
        "lengths": lengths,
        "nonamp_probs": nonamp_probs,
        "nonamp_mask": nonamp_mask
    }

# -------------------- Overall NON-AMP summary --------------------
summary_rows = []

for name, d in cache.items():
    X = d["X"]
    nonamp_probs = d["nonamp_probs"]
    nonamp_mask = d["nonamp_mask"]

    n_nonamp = nonamp_mask.sum()
    nonamp_yield = n_nonamp / len(X)

    row = {
        "Model": name,
        "Total Peptides": len(X),
        "Non-AMP-positive": n_nonamp,
        "Non-AMP Yield (0.5)": round(nonamp_yield, 3),
        "Embedding Diversity (All)": round(embedding_diversity(X), 3),
        "Embedding Diversity (Non-AMPs)": round(
            embedding_diversity(X[nonamp_mask]) if n_nonamp > 0 else np.nan, 3
        ),
    }

    for t in NONAMP_THRESHOLDS:
        count = (nonamp_probs >= t).sum()
        row[f"NON-AMP@{t}_count"] = count
        row[f"NON-AMP@{t}_yield"] = round(count / len(X), 3)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).sort_values("Model")
print("\n=== NON-AMP Overall Summary ===")
print(summary_df.to_string(index=False))

# -------------------- Length-based NON-AMP summary --------------------
length_rows = []

for name, d in cache.items():
    X = d["X"]
    lengths = d["lengths"]
    nonamp_mask = d["nonamp_mask"]

    for l in np.unique(lengths):
        mask_l = lengths == l
        n = mask_l.sum()
        if n == 0:
            continue

        nonamp_l = nonamp_mask[mask_l]

        length_rows.append({
            "Model": name,
            "Peptide Length": l,
            "Total Peptides": n,
            "Non-AMP-positive": nonamp_l.sum(),
            "Non-AMP Yield": round(nonamp_l.mean(), 3),
            "Embedding Diversity (Non-AMPs)": round(
                embedding_diversity(X[mask_l][nonamp_l]) if nonamp_l.any() else np.nan, 3
            )
        })

length_df = pd.DataFrame(length_rows).sort_values(["Model", "Peptide Length"])
print("\n=== NON-AMP Length-based Summary ===")
print(length_df.to_string(index=False))
