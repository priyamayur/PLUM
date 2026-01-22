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
MIC_CLASSIFIER_PATH = "./saved_models_MIC/best_model/classifier.pkl"

AMP_THRESHOLD = 0.5

fasta_files = {
    "AMPGAN": "./data/AMPGAN_generated_AMP_45k.fasta",
    "DeanVAE": "./data/DeanVAE_generated_AMP.fasta",
    "HydrAMP": "./data/HydrAMP_generated_AMP_45k.fasta",
    "MullerRNN": "./data/MullerRNN_generated_AMP.fasta",
    "PLUM": "./data/PLUM_generated_AMP_func1_45k.fasta",
    "Baseline1": "./data/Baseline1_generated_AMP_func1_45k.fasta",
    "Baseline2": "./data/Baseline2_generated_AMP_func1_45k.fasta",
}

# -------------------- Load models once --------------------
classifier = joblib.load(CLASSIFIER_PATH)
mic_classifier = joblib.load(MIC_CLASSIFIER_PATH)

with open(ENCODER_INFO_PATH) as f:
    encoder_name = json.load(f)["encoder_name"]

encoder = T5EncoderModel.from_pretrained(encoder_name).to(DEVICE).eval()
tokenizer = T5Tokenizer.from_pretrained(encoder_name, do_lower_case=False)

# -------------------- Helper functions --------------------
def generate_embeddings(sequences, model, tokenizer, batch_size=64):
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = [" ".join(seq) for seq in sequences[i:i + batch_size]]
        tokens = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            out = model(**tokens).last_hidden_state
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


# -------------------- Embed & cache once --------------------
cache = {}

for name, fasta in fasta_files.items():
    records = list(SeqIO.parse(fasta, "fasta"))
    seqs = [str(r.seq).upper() for r in records]

    if not seqs:
        continue

    print(f"⏳ Embedding {name} ({len(seqs)} peptides)")
    X = generate_embeddings(seqs, encoder, tokenizer)
    lengths = np.array([len(s) for s in seqs])

    amp_probs = classifier.predict_proba(X)[:, 1]
    amp_mask = amp_probs >= AMP_THRESHOLD

    cache[name] = {
        "X": X,
        "lengths": lengths,
        "amp_mask": amp_mask
    }

# -------------------- Overall summary --------------------
summary = []

for name, d in cache.items():
    X = d["X"]
    amp_mask = d["amp_mask"]

    num_amp = amp_mask.sum()
    amp_yield = num_amp / len(X)

    if num_amp > 0:
        mic_preds = mic_classifier.predict(X[amp_mask])
        mic_yield = (mic_preds == 1).mean()
        div_amp = embedding_diversity(X[amp_mask])
    else:
        mic_yield = 0.0
        div_amp = np.nan

    div_all = embedding_diversity(X)

    summary.append({
        "Model": name,
        "Total Peptides": len(X),
        "AMP Yield": round(amp_yield, 3),
        "Potent MIC Yield": round(mic_yield, 3),
        "Embedding Diversity (AMPs)": round(div_amp, 3),
        "Embedding Diversity (All)": round(div_all, 3),
    })

summary_df = pd.DataFrame(summary).sort_values("Model")
print("\n=== Overall Summary ===")
print(summary_df.to_string(index=False))

# -------------------- Length-based summary --------------------
length_rows = []

for name, d in cache.items():
    X = d["X"]
    lengths = d["lengths"]
    amp_mask = d["amp_mask"]

    for l in np.unique(lengths):
        mask_l = lengths == l
        n = mask_l.sum()
        if n == 0:
            continue

        amp_l = amp_mask[mask_l]
        amp_yield = amp_l.mean()

        if amp_l.any():
            mic_preds = mic_classifier.predict(X[mask_l][amp_l])
            mic_yield = (mic_preds == 1).mean()
            div_amp = embedding_diversity(X[mask_l][amp_l])
        else:
            mic_yield = 0.0
            div_amp = np.nan

        length_rows.append({
            "Model": name,
            "Peptide Length": l,
            "Total Peptides": n,
            "AMP Yield": round(amp_yield, 3),
            "Potent MIC Yield": round(mic_yield, 3),
            "Embedding Diversity (AMPs)": round(div_amp, 3),
        })

length_df = pd.DataFrame(length_rows).sort_values(["Model", "Peptide Length"])
print("\n=== Length-based Summary ===")
print(length_df.to_string(index=False))
