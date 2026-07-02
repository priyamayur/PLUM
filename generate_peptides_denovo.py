#!/usr/bin/env python3
import argparse
from datetime import datetime
import time
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import trange

from training_generative_model.generative_model_id_004 import (
    PeptideCSVAE_LSTM, AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, START_TOKEN, STOP_TOKEN,
    length_to_bin, LENGTH_BINS, NUM_LENGTH_BINS
)


# ----------------------------- Helper Functions -----------------------------
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int):
    """Set seed for reproducible peptide generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # These make CUDA behavior more reproducible.
    # They can slightly reduce speed, but usually not meaningfully for generation.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[{now()}] 🌱 Seed set to: {seed}")


def sample_target_lengths(n_samples, min_len=5, max_len=35):
    """
    Sample target lengths for each peptide.

    If n_samples <= number of possible lengths, every sampled length is unique.
    If n_samples > number of possible lengths, lengths repeat only after all
    lengths have been used once.
    """
    possible_lengths = np.arange(min_len, max_len + 1)
    n_lengths = len(possible_lengths)

    sampled_lengths = []

    full_cycles = n_samples // n_lengths
    remainder = n_samples % n_lengths

    for _ in range(full_cycles):
        sampled_lengths.extend(np.random.permutation(possible_lengths).tolist())

    if remainder > 0:
        sampled_lengths.extend(
            np.random.choice(possible_lengths, size=remainder, replace=False).tolist()
        )

    sampled_lengths = np.array(sampled_lengths, dtype=int)
    np.random.shuffle(sampled_lengths)

    return sampled_lengths


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

    print(f"[{now()}] ✅ Model loaded.")
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
    device="cpu"
):
    """
    Generate peptides conditioned on one target function and one exact target length.

    Important:
    - Each peptide gets its own z.
    - Each peptide gets its own sampled w from p(w | function).
    - Each peptide gets its own sampled v from p(v | length).
    - Batch generation is still parallel and fast.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    bins = np.asarray(length_to_bin(target_length))
    if bins.size == 0:
        raise ValueError(f"No length bin found for target length {target_length}")

    target_bin_idx = int(bins[0])

    batch_size = n_samples_per_condition

    # -----------------------------
    # Conditioning
    # -----------------------------
    y_func = torch.tensor(
        [[target_function]],
        dtype=torch.float32,
        device=device
    ).repeat(batch_size, 1)

    y_len = F.one_hot(
        torch.tensor([target_bin_idx], dtype=torch.long, device=device),
        NUM_LENGTH_BINS
    ).float().repeat(batch_size, 1)

    # -----------------------------
    # Latent sampling
    # -----------------------------
    # Each peptide gets a different residual sequence latent.
    z_batch = torch.randn(batch_size, model.z_dim, device=device)

    # Priors are expanded across batch, so each peptide gets its own epsilon.
    mu_w, logvar_w = model.p_w_prior(y_func)
    w_batch = model.reparameterize(mu_w, logvar_w)

    mu_v, logvar_v = model.p_v_prior(y_len)
    v_batch = model.reparameterize(mu_v, logvar_v)

    # -----------------------------
    # LSTM initialization
    # -----------------------------
    h = torch.zeros(
        model.lstm_layers,
        batch_size,
        model.hidden_dim,
        device=device
    )

    c = torch.zeros(
        model.lstm_layers,
        batch_size,
        model.hidden_dim,
        device=device
    )

    start_idx = torch.full(
        (batch_size,),
        AA_TO_IDX[START_TOKEN],
        dtype=torch.long,
        device=device
    )

    input_t = F.one_hot(
        start_idx,
        num_classes=model.input_dim
    ).float().unsqueeze(1)

    seqs = [""] * batch_size
    finished = np.zeros(batch_size, dtype=bool)

    start_time = time.time()

    for step in trange(
        target_length,
        desc=f"Generating length {target_length}",
        leave=False
    ):
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
            input_idx = topk_idx[torch.arange(batch_size, device=device), sampled]
        else:
            probs = F.softmax(logits, dim=1)
            input_idx = torch.multinomial(probs, 1).squeeze(1)

        input_t = F.one_hot(
            input_idx,
            num_classes=model.input_dim
        ).float().unsqueeze(1)

        input_idx_cpu = input_idx.detach().cpu().numpy()

        for i in range(batch_size):
            if finished[i]:
                continue

            aa = IDX_TO_AA[input_idx_cpu[i]]

            # For exact-length generation, ignore special tokens.
            # We only append real amino acids until target_length is reached.
            if aa in (PAD_TOKEN, START_TOKEN, STOP_TOKEN):
                continue

            if len(seqs[i]) < target_length:
                seqs[i] += aa

            if len(seqs[i]) >= target_length:
                finished[i] = True

        if finished.all():
            break

    elapsed = time.time() - start_time

    valid_peptides = [
        s for s in seqs
        if min_len <= len(s) <= max_len
    ]

    print(
        f"[{now()}] ⏱ Finished length {target_length}: "
        f"{len(valid_peptides)}/{batch_size} valid peptides in {elapsed:.2f}s."
    )

    return valid_peptides


# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate de novo peptides conditioned on target function and length"
    )

    parser.add_argument(
        "-f", "--target_function",
        type=int,
        required=True,
        help="Target function ID. Example: 1 for AMP, 0 for non-AMP."
    )

    parser.add_argument(
        "-l", "--target_length",
        type=int,
        default=None,
        help="Target peptide length. If omitted, each peptide gets a random length from 5-35."
    )

    parser.add_argument(
        "-n", "--n_samples",
        type=int,
        default=10,
        help="Number of peptides to generate. Default: 10."
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="generated_peptides",
        help="Output directory to save peptides TSV. Default: ./generated_peptides/"
    )

    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If omitted, a time-based seed is used."
    )

    args = parser.parse_args()

    # -----------------------------
    # Seed
    # -----------------------------
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % 100000
    set_seed(seed)

    min_len = 5
    max_len = 35

    # -----------------------------
    # Target length setup
    # -----------------------------
    if args.target_length is None:
        target_lengths = sample_target_lengths(
            n_samples=args.n_samples,
            min_len=min_len,
            max_len=max_len
        )

        print(
            f"[{now()}] 🎲 No target length provided. "
            f"Randomly assigned one target length per peptide."
        )

        unique_lengths, counts = np.unique(target_lengths, return_counts=True)

        print(f"[{now()}] Length distribution:")
        for length, count in zip(unique_lengths, counts):
            print(f"  Length {length}: {count}")

    else:
        if not (min_len <= args.target_length <= max_len):
            raise ValueError(
                f"target_length must be between {min_len} and {max_len}. "
                f"Received {args.target_length}."
            )

        target_lengths = np.array([args.target_length] * args.n_samples, dtype=int)

        print(
            f"[{now()}] Using provided target_length = {args.target_length}. "
            f"All peptides will target this length."
        )

    # -----------------------------
    # Paths
    # -----------------------------
    dir_path = Path(__file__).resolve().parent

    output_path = dir_path / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{now()}] Device: {device}")

    checkpoint_path = (
        dir_path
        / "trained_models"
        / "generative_model"
        / "PLUM_new_analysis_renew_part4_v2_004.pth"
    )

    # -----------------------------
    # Load model
    # -----------------------------
    model = load_model(checkpoint_path, device)

    print(
        f"[{now()}] 🧪 Generating {args.n_samples} peptides "
        f"with function {args.target_function}..."
    )

    temperature = 1.0
    top_k = 0

    # -----------------------------
    # Generate grouped by target length
    # -----------------------------
    all_rows = []
    peptide_counter = 0

    unique_lengths, counts = np.unique(target_lengths, return_counts=True)

    for target_length, count in zip(unique_lengths, counts):
        print(
            f"[{now()}] -> Generating {count} peptide(s) "
            f"with function {args.target_function}, length {target_length}"
        )

        peptides_for_length = generate_peptides_denovo(
            model=model,
            target_function=args.target_function,
            target_length=int(target_length),
            n_samples_per_condition=int(count),
            min_len=min_len,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            device=device
        )

        for peptide in peptides_for_length:
            peptide_counter += 1
            all_rows.append({
                "ID": f"peptide_{peptide_counter}",
                "Peptide": peptide,
                "Target_Function": args.target_function,
                "Target_Length": int(target_length)
            })

    print(
        f"[{now()}] ✅ Generated {len(all_rows)} valid peptides "
        f"out of requested {args.n_samples}."
    )

    # -----------------------------
    # Save output
    # -----------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.target_length is None:
        output_file = (
            output_path
            / f"peptides_func{args.target_function}_random_lengths_seed{seed}_{timestamp}.tsv"
        )
    else:
        output_file = (
            output_path
            / f"peptides_func{args.target_function}_len{args.target_length}_seed{seed}_{timestamp}.tsv"
        )

    df = pd.DataFrame(all_rows)

    df.to_csv(output_file, sep="\t", index=False)

    print(f"[{now()}] ✅ Saved peptides to {output_file}")


if __name__ == "__main__":
    main()