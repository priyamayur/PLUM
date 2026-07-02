#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
from pathlib import Path
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
    amp_path = dir_path / "trained_models" / "classifiers" / "AMP_classifier" / "classifier.pkl"
    apm_path = dir_path / "trained_models" / "classifiers" / "Antibacterial_Potency_Model" / "classifier.pkl"
    amp_clf = joblib.load(amp_path)
    apm_clf = joblib.load(apm_path)
    print("✅ Classifiers loaded.")
    return amp_clf, apm_clf

def load_encoder(dir_path, device):
    """Load ProtT5 encoder and tokenizer from current directory."""
    print("🔄 Loading ProtT5 encoder and tokenizer...")
    encoder_info_path = dir_path / "trained_models" / "classifiers" / "AMP_classifier" / "encoder_info.json"
    with encoder_info_path.open() as f:
        encoder_info = json.load(f)
    encoder_name = encoder_info["encoder_name"]
    encoder = T5EncoderModel.from_pretrained(encoder_name).to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(encoder_name, do_lower_case=False)
    print(f"✅ ProtT5 encoder '{encoder_name}' loaded.")
    return encoder, tokenizer

def load_fasta(fasta_path):
    """Load sequences from FASTA file."""
    fasta_path = Path(fasta_path)
    print(f"🔄 Loading sequences from {fasta_path}...")
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
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

def classify_amp_apm(embeddings, amp_clf, apm_clf, amp_threshold=0.5, apm_threshold=0.5):
    """Classify sequences as AMP and Antibacterial Potency."""
    print("🔄 Predicting AMP activity...")

    amp_probs = np.round(amp_clf.predict_proba(embeddings)[:, 1], 4)
    amp_positive_mask = amp_probs >= amp_threshold
    amp_classes = np.where(amp_positive_mask, "AMP", "Non-AMP")

    print("✅ AMP prediction done.")

    print("🔄 Predicting Antibacterial Potency for AMP-positive sequences...")

    apm_probs = np.full(len(embeddings), np.nan)
    apm_classes = np.full(len(embeddings), "NA", dtype=object)

    if np.any(amp_positive_mask):
        X_amp = embeddings[amp_positive_mask]

        apm_probs_pos = np.round(apm_clf.predict_proba(X_amp)[:, 1], 4)
        apm_classes_pos = np.where(
            apm_probs_pos >= apm_threshold,
            "Antibacterial_active",
            "Antibacterial_inactive"
        )

        apm_probs[amp_positive_mask] = apm_probs_pos
        apm_classes[amp_positive_mask] = apm_classes_pos

    print("✅ Antibacterial Potency prediction done.")

    return amp_probs, amp_classes, apm_probs, apm_classes

def save_results(records, sequences, amp_probs, amp_classes, apm_probs, apm_classes, output_path):
    """Save results to TSV file."""
    print(f"🔄 Saving predictions to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "seq_id": [rec.id for rec in records],
        "sequence": sequences,
        "amp_prob": amp_probs,
        "amp_class": amp_classes,
        "antibacterial_potency_prob": apm_probs,
        "antibacterial_potency_class": apm_classes
    })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / ("classified_peptides_" + timestamp + ".tsv")
    df.to_csv(output_file, sep="\t", index=False)
    print(f"✅ Predictions saved to {output_file}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Predict AMP and Antibacterial Potency activity for peptide sequences")
    parser.add_argument("--input", "-i", required=True, help="Input FASTA file path")
    parser.add_argument("--amp_threshold", type=float, default=0.5, help="Threshold for AMP classification (default: 0.5)")
    parser.add_argument("--antibacterial_potency_threshold", type=float, default=0.5, help="Threshold for Antibacterial Potency classification (default: 0.5)")
    parser.add_argument("-o", "--output_dir", default=None, help="Output directory to save peptides TSV (default: ./classified_peptides/)")

    args = parser.parse_args()
    
    dir_path = Path(__file__).resolve().parent  # Script directory as base path
    
    output_dir = args.output_dir or "classified_peptides"
    output_path = dir_path / output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_threshold = args.amp_threshold
    apm_threshold = args.antibacterial_potency_threshold
    batch_size = 64
    
    amp_clf, apm_clf = load_classifiers(dir_path)
    encoder, tokenizer = load_encoder(dir_path, device)

    records, sequences = load_fasta(args.input)

    embeddings = generate_embeddings(sequences, encoder, tokenizer, device, batch_size=batch_size)

    amp_probs, amp_classes, apm_probs, apm_classes = classify_amp_apm(
        embeddings, amp_clf, apm_clf, amp_threshold=amp_threshold, apm_threshold=apm_threshold
    )

    save_results(records, sequences, amp_probs, amp_classes, apm_probs, apm_classes, output_path)

if __name__ == "__main__":
    main()
