from Bio import SeqIO
import joblib
import torch
import numpy as np
import json
from transformers import T5EncoderModel, T5Tokenizer
import pandas as pd

# -------------------- Settings --------------------
dir_path = "/content/drive/MyDrive/Colab Notebooks/PLUM/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AMP_MODEL_DIR = dir_path + "trained_models/classifiers/"
AMP_CLASSIFIER_PATH = f"{AMP_MODEL_DIR}/AMP_classifier/classifier.pkl"
AMP_ENCODER_INFO_PATH = f"{AMP_MODEL_DIR}/AMP_classifier/encoder_info.json"
MIC_CLASSIFIER_PATH = f"{AMP_MODEL_DIR}/MIC_classifier/classifier.pkl"

INPUT_FASTA = dir_path + "data/peptide_test.fasta"
OUTPUT_TSV = dir_path + "predictions.tsv"

AMP_THRESHOLD = 0.5

# -------------------- Load classifiers --------------------
amp_classifier = joblib.load(AMP_CLASSIFIER_PATH)
mic_classifier = joblib.load(MIC_CLASSIFIER_PATH)

# -------------------- Load ProtT5 Encoder --------------------
with open(AMP_ENCODER_INFO_PATH) as f:
    encoder_info = json.load(f)
encoder_name = encoder_info["encoder_name"]

encoder = T5EncoderModel.from_pretrained(encoder_name).to(DEVICE).eval()
tokenizer = T5Tokenizer.from_pretrained(encoder_name, do_lower_case=False)

# -------------------- Load FASTA --------------------
input_records = list(SeqIO.parse(INPUT_FASTA, "fasta"))
input_sequences = [str(rec.seq).upper() for rec in input_records]

if not input_sequences:
    raise ValueError(f"No sequences found in {INPUT_FASTA}")

# -------------------- Generate ProtT5 embeddings --------------------
def generate_embeddings(sequences, model, tokenizer, batch_size=64):
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = [" ".join(list(seq)) for seq in sequences[i:i+batch_size]]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**tokens)
            hidden_states = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            masked_embeddings = hidden_states * attention_mask
            seq_lengths = attention_mask.sum(dim=1)
            pooled = masked_embeddings.sum(dim=1) / seq_lengths
        all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)

embeddings = generate_embeddings(input_sequences, encoder, tokenizer)

# -------------------- Classify AMP --------------------
amp_probs = amp_classifier.predict_proba(embeddings)[:, 1]
amp_positive_mask = amp_probs >= AMP_THRESHOLD
amp_classes = np.where(amp_positive_mask, "AMP", "Non-AMP")

# -------------------- Classify MIC (only AMP-positive) --------------------
mic_probs = np.full(len(input_sequences), np.nan)
mic_classes = np.full(len(input_sequences), "NA", dtype=object)

if np.any(amp_positive_mask):
    X_amp = embeddings[amp_positive_mask]
    mic_probs_pos = mic_classifier.predict_proba(X_amp)[:, 1]
    mic_classes_pos = np.where(mic_probs_pos >= 0.5, "MIC_active", "MIC_inactive")
    mic_probs[amp_positive_mask] = mic_probs_pos
    mic_classes[amp_positive_mask] = mic_classes_pos

# -------------------- Save results to TSV --------------------
results_df = pd.DataFrame({
    "seq_id": [rec.id for rec in input_records],
    "sequence": input_sequences,
    "amp_prob": amp_probs,
    "amp_class": amp_classes,
    "mic_prob": mic_probs,
    "mic_class": mic_classes
})

results_df.to_csv(OUTPUT_TSV, sep="\t", index=False)
print(f"✅ Predictions saved to {OUTPUT_TSV}")
