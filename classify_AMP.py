#!/usr/bin/env python3
import argparse
import os
from Bio import SeqIO
import joblib
import torch
import numpy as np
import json
from transformers import T5EncoderModel, T5Tokenizer
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# -------------------- Functions --------------------

def load_classifiers(dir_path):
    """Load AMP and MIC classifiers from current directory."""
    print("🔄 Loading AMP and MIC classifiers...")
    amp_path = os.path.join(dir_path, "trained_models/classifiers/AMP_classifier/classifier.pkl")
    mic_path = os.path.join(dir_path, "trained_models/classifiers/AMP_MIC_classifier/classifier.pkl")
    amp_clf = joblib.load(amp_path)
    mic_clf = joblib.load(mic_path)
    print("✅ Classifiers loaded.")
    return amp_clf, mic_clf

def load_encoder(dir_path, device):
    """Load ProtT5 encoder and tokenizer from current directory."""
    print("🔄 Loading ProtT5 encoder and tokenizer...")
    encoder_info_path = os.path.join(dir_path, "trained_models/classifiers/AMP_classifier/encoder_info.json")
    with open(encoder_info_path) as f:
        encoder_info = json.load(f)
    encoder_name = encoder_info["encoder_name"]
    encoder = T5EncoderModel.from_pretrained(encoder_name).to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(encoder_name, do_lower_case=False)
    print(f"✅ ProtT5 encoder '{encoder_name}' loaded.")
    return encoder, tokenizer

def load_fasta(fasta_path):
    """Load sequences from FASTA file."""
    print(f"🔄 Loading sequences from {fasta_path}...")
    records = list(SeqIO.parse(fasta_path, "fasta"))
    sequences = [str(rec.seq).upper() for rec in records]
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")
    print(f"✅ Loaded {len(sequences)} sequences.")
    return records, sequences

def generate_embeddings(sequences, model, tokenizer, device, batch_size=64):
    """Generate ProtT5 embeddings for a list of sequences."""
    print("🔄 Generating ProtT5 embeddings...")
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding batches"):
        batch = [" ".join(list(seq)) for seq in sequences[i:i+batch_size]]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**tokens)
            hidden_states = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"].unsqueeze(-1)
            masked_embeddings = hidden_states * attention_mask
            seq_lengths = attention_mask.sum(dim=1)
            pooled = masked_embeddings.sum(dim=1) / seq_lengths
        all_embeddings.append(pooled.cpu().numpy())
    print("✅ Embeddings generated.")
    return np.vstack(all_embeddings)

def classify_amp_mic(embeddings, amp_clf, mic_clf, amp_threshold=0.5):
    """Classify sequences as AMP and MIC."""
    print("🔄 Predicting AMP activity...")
    amp_probs = amp_clf.predict_proba(embeddings)[:, 1]
    amp_positive_mask = amp_probs >= amp_threshold
    amp_classes = np.where(amp_positive_mask, "AMP", "Non-AMP")
    print(f"✅ AMP prediction done. {amp_positive_mask.sum()} sequences predicted as AMP.")

    print("🔄 Predicting MIC activity for AMP-positive sequences...")
    mic_probs = np.full(len(embeddings), np.nan)
    mic_classes = np.full(len(embeddings), "NA", dtype=object)

    if np.any(amp_positive_mask):
        X_amp = embeddings[amp_positive_mask]
        mic_probs_pos = mic_clf.predict_proba(X_amp)[:, 1]
        mic_classes_pos = np.where(mic_probs_pos >= 0.5, "MIC_active", "MIC_inactive")
        mic_probs[amp_positive_mask] = mic_probs_pos
        mic_classes[amp_positive_mask] = mic_classes_pos
    print("✅ MIC prediction done.")

    return amp_probs, amp_classes, mic_probs, mic_classes

def save_results(records, sequences, amp_probs, amp_classes, mic_probs, mic_classes, output_path):
    """Save results to TSV file."""
    print(f"🔄 Saving predictions to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame({
        "seq_id": [rec.id for rec in records],
        "sequence": sequences,
        "amp_prob": amp_probs,
        "amp_class": amp_classes,
        "mic_prob": mic_probs,
        "mic_class": mic_classes
    })
    df.to_csv(output_path, sep="\t", index=False)
    print(f"✅ Predictions saved to {output_path}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Predict AMP and MIC activity for peptide sequences")
    parser.add_argument("--input", "-i", required=True, help="Input FASTA file path")
    parser.add_argument("--output", "-o", required=True, help="Output TSV file path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--amp_threshold", type=float, default=0.5, help="Threshold for AMP classification")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding generation")

    args = parser.parse_args()
    device = torch.device(args.device)

    dir_path = os.getcwd()  # Current directory as base path

    # Load models
    amp_clf, mic_clf = load_classifiers(dir_path)
    encoder, tokenizer = load_encoder(dir_path, device)

    # Load sequences
    records, sequences = load_fasta(args.input)

    # Generate embeddings
    embeddings = generate_embeddings(sequences, encoder, tokenizer, device, batch_size=args.batch_size)

    # Classify
    amp_probs, amp_classes, mic_probs, mic_classes = classify_amp_mic(
        embeddings, amp_clf, mic_clf, amp_threshold=args.amp_threshold
    )

    # Save results
    save_results(records, sequences, amp_probs, amp_classes, mic_probs, mic_classes, args.output)

if __name__ == "__main__":
    main()
