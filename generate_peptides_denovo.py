#!/usr/bin/env python3
import argparse
from datetime import datetime
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import trange
from training_generative_model.generative_model import (
    PeptideCSVAE_LSTM, AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, START_TOKEN,
    length_to_bin, LENGTH_BINS, NUM_LENGTH_BINS
)

# ----------------------------- Functions -----------------------------
def load_model(checkpoint_path, device):
    """Load the pre-trained generative model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["model_config"]
    model = PeptideCSVAE_LSTM(
        seq_len=cfg["seq_len"],
        z_dim=cfg["z_dim"],
        w_dim=cfg["w_dim"],
        v_dim=cfg["v_dim"],
        hidden_dim=cfg["hidden_dim"],
        cond_dim=cfg["cond_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✅ Model loaded.")
    return model

def generate_peptides_denovo(
    model,
    target_function,
    target_length,
    n_samples_per_condition=1,
    min_len=5,
    max_len=35,
    temperature=1.0,
    top_k=0,
    device='cpu'
):
    """Generate peptides conditioned on a target function and exact target length."""
    device = torch.device(device)
    model.to(device)
    model.eval()

    bins = np.asarray(length_to_bin(target_length))
    if bins.size == 0:
        raise ValueError(f"No length bin found for target length {target_length}")
    target_bin_idx = int(bins[0])

    y_func = torch.tensor([[target_function]], dtype=torch.float32, device=device)
    y_len = F.one_hot(
        torch.tensor([target_bin_idx], dtype=torch.long),
        NUM_LENGTH_BINS
    ).float().to(device)

    z = torch.randn(1, model.z_dim, device=device)

    mu_w, logvar_w = model.p_w_prior(y_func)
    w = model.reparameterize(mu_w, logvar_w)
    if w.dim() == 1:
        w = w.unsqueeze(0)

    mu_v, logvar_v = model.p_v_prior(y_len)
    v = model.reparameterize(mu_v, logvar_v)
    if v.dim() == 1:
        v = v.unsqueeze(0)

    z_batch = z.repeat(n_samples_per_condition, 1)
    w_batch = w.repeat(n_samples_per_condition, 1)
    v_batch = v.repeat(n_samples_per_condition, 1)

    h = torch.zeros(
        model.lstm_layers,
        n_samples_per_condition,
        model.hidden_dim,
        device=device
    )
    c = torch.zeros(
        model.lstm_layers,
        n_samples_per_condition,
        model.hidden_dim,
        device=device
    )

    start_idx = torch.full(
        (1,),
        AA_TO_IDX[START_TOKEN],
        dtype=torch.long,
        device=device
    )
    input_t = F.one_hot(
        start_idx,
        num_classes=model.input_dim
    ).float().unsqueeze(1)
    input_t = input_t.repeat(n_samples_per_condition, 1, 1)

    seqs = [''] * n_samples_per_condition
    finished = [False] * n_samples_per_condition

    start_time = time.time()
    for step in trange(target_length, desc="Generating peptides"):
        lstm_input = torch.cat(
            [
                input_t,
                z_batch.unsqueeze(1),
                w_batch.unsqueeze(1),
                v_batch.unsqueeze(1)
            ],
            dim=2
        )
        out, (h, c) = model.decoder_lstm(lstm_input, (h, c))
        logits = model.out_x(out).squeeze(1)
        logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=1)
            probs = F.softmax(topk_vals, dim=1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            input_idx = topk_idx[torch.arange(n_samples_per_condition), sampled]
        else:
            probs = F.softmax(logits, dim=1)
            input_idx = torch.multinomial(probs, 1).squeeze(1)

        input_t = F.one_hot(
            input_idx,
            num_classes=model.input_dim
        ).float().unsqueeze(1)

        for i, idx in enumerate(input_idx):
            if finished[i]:
                continue
            aa = IDX_TO_AA[idx.item()]
            if aa in (PAD_TOKEN, START_TOKEN):
                continue
            seqs[i] += aa
            if len(seqs[i]) >= target_length:
                finished[i] = True

        if all(finished):
            break

    elapsed = time.time() - start_time
    print(f"⏱  Finished generating {n_samples_per_condition} peptides in {elapsed:.2f}s.")
    return [s for s in seqs if len(s) >= min_len]

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate de novo peptides conditioned on target function and length"
    )
    parser.add_argument(
        "-f", "--target_function",
        type=int, required=True,
        help="Target function ID (required)"
    )
    parser.add_argument(
        "-l", "--target_length",
        type=int, default=None,
        help="Target peptide length (optional, random 5-35 if not provided)"
    )
    parser.add_argument(
        "-n", "--n_samples",
        type=int, default=10,
        help="Number of peptides to generate (default: 10)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str, default="generated_peptides",
        help="Output directory to save peptides CSV (default: ./generated_peptides/)"
    )

    args = parser.parse_args()

    if args.target_length is None:
        target_length = np.random.randint(5, 36)
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"🎲 No target length provided. Randomly selected target_length = {target_length}"
        )
    else:
        target_length = args.target_length
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Using provided target_length = {target_length}"
        )

    dir_path = Path(__file__).resolve().parent

    output_dir = args.output_dir
    output_path = dir_path / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Device: {device}")

    checkpoint_path = (
        dir_path
        / "trained_models"
        / "generative_model"
        / "PLUM_checkpoint_v5.pth"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]

    model = PeptideCSVAE_LSTM(
        seq_len=model_cfg["seq_len"],
        z_dim=model_cfg["z_dim"],
        w_dim=model_cfg["w_dim"],
        v_dim=model_cfg["v_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        cond_dim=model_cfg["cond_dim"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Model loaded.")

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"🧪 Generating {args.n_samples} peptides with function "
        f"{args.target_function} and length {target_length}..."
    )

    temperature = 1.0
    top_k = 0

    peptides = generate_peptides_denovo(
        model,
        target_function=args.target_function,
        target_length=target_length,
        n_samples_per_condition=args.n_samples,
        temperature=temperature,
        top_k=top_k,
        device=device
    )

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"✅ Generated {len(peptides)} peptides."
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        output_path
        / f"peptides_func{args.target_function}_len{target_length}_{timestamp}.tsv"
    )

    df = pd.DataFrame({
        "ID": [f"peptide_{i+1}" for i in range(len(peptides))],
        "Peptide": peptides,
        "Function": [args.target_function] * len(peptides),
        "Length": [len(p) for p in peptides]
    })

    df.to_csv(output_file, sep="\t", index=False)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"✅ Saved peptides to {output_file}"
    )

if __name__ == "__main__":
    main()
