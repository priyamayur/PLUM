#!/usr/bin/env python3

import argparse
import time
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

from training_generative_model.generative_model import (
    PeptideCSVAE_LSTM,
    AA_TO_IDX,
    IDX_TO_AA,
    PAD_TOKEN,
    START_TOKEN,
    STOP_TOKEN,
    length_to_bin,
    LENGTH_BINS,
    NUM_LENGTH_BINS
)


# -------------------- Utilities --------------------

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def set_seed(seed: int):
    """Set seed for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log(f"🌱 Seed set to: {seed}")


def clean_sequence(seq):
    """Remove special tokens and force uppercase."""
    return (
        str(seq)
        .upper()
        .replace(PAD_TOKEN, "")
        .replace(START_TOKEN, "")
        .replace(STOP_TOKEN, "")
        .replace("Z", "")
    )


def get_length_bin_idx(target_length):
    """Return length-bin index for an exact target length."""
    bins = np.asarray(length_to_bin(target_length))

    if bins.size == 0:
        raise ValueError(f"No length bin found for target length {target_length}")

    return int(bins[0])


# -------------------- Model --------------------

def load_model(checkpoint_path, device):
    """Load pre-trained PLUM generative model."""
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

    log("✅ Model loaded.")
    return model


# -------------------- Priors --------------------

def build_prior_cache(model, device):
    """Cache all function and length-bin priors once."""
    prior_cache = {}

    for func in [0, 1]:
        y_func = torch.tensor(
            [[func]],
            dtype=torch.float32,
            device=device
        )

        mu_w, logvar_w = model.p_w_prior(y_func)
        prior_cache[("w", func)] = (mu_w, logvar_w)

    for len_bin in range(NUM_LENGTH_BINS):
        y_len = F.one_hot(
            torch.tensor([len_bin], dtype=torch.long, device=device),
            NUM_LENGTH_BINS
        ).float()

        mu_v, logvar_v = model.p_v_prior(y_len)
        prior_cache[("v", len_bin)] = (mu_v, logvar_v)

    return prior_cache


# -------------------- Latent Sampling --------------------

def sample_latents(
    model,
    mu_z,
    n_samples,
    target_func,
    target_len_idx,
    device,
    perturb_std=0.01,
    prior_cache=None,
    stochastic=True
):
    """
    Sample prototype-guided latent vectors.

    z: centered on prototype latent with independent perturbation
    w: independently sampled from target-function prior
    v: independently sampled from target-length prior
    """

    # Prototype-centered residual sequence latent
    z_proto = mu_z.repeat(n_samples, 1)

    if stochastic:
        z = z_proto + torch.randn_like(z_proto) * perturb_std
    else:
        z = z_proto

    # Function prior
    mu_w, logvar_w = prior_cache[("w", target_func)]
    mu_w_batch = mu_w.repeat(n_samples, 1)
    logvar_w_batch = logvar_w.repeat(n_samples, 1)

    # Length prior
    mu_v, logvar_v = prior_cache[("v", target_len_idx)]
    mu_v_batch = mu_v.repeat(n_samples, 1)
    logvar_v_batch = logvar_v.repeat(n_samples, 1)

    if stochastic:
        w = model.reparameterize(mu_w_batch, logvar_w_batch)
        v = model.reparameterize(mu_v_batch, logvar_v_batch)
    else:
        w = mu_w_batch
        v = mu_v_batch

    return z, w, v


# -------------------- Sequence Decoding --------------------

def decode_sequence_batch(
    model,
    z,
    w,
    v,
    target_len,
    proto_onehot=None,
    beta=0.45,
    min_len=5,
    max_len=35,
    temperature=1.0,
    device="cpu",
    stochastic=True,
    top_k=0
):
    """
    Decode a batch of prototype-guided peptides toward one target length.
    """

    batch_size = z.size(0)

    seqs = [""] * batch_size

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device
    )

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

    stop_idx = AA_TO_IDX[STOP_TOKEN]
    pad_idx = AA_TO_IDX[PAD_TOKEN]

    for t in range(target_len):
        lstm_input = torch.cat(
            [
                input_t,
                z.unsqueeze(1),
                w.unsqueeze(1),
                v.unsqueeze(1)
            ],
            dim=2
        )

        out, (h, c) = model.decoder_lstm(lstm_input, (h, c))
        logits = model.out_x(out).squeeze(1)
        logits = logits / max(temperature, 1e-6)

        # Finished sequences should only emit PAD
        logits[finished, :] = -1e9
        logits[finished, pad_idx] = 0

        # Do not allow early STOP before minimum length
        if t < min_len:
            logits[:, stop_idx] = -1e9

        # Force STOP at final decoding step
        if t == target_len - 1:
            logits[:, :] = -1e9
            logits[:, stop_idx] = 0

        probs = F.softmax(logits, dim=1)

        # Prototype-anchored decoding
        if proto_onehot is not None and t < proto_onehot.size(1) and beta > 0:
            probs = beta * proto_onehot[:, t, :] + (1 - beta) * probs

        if stochastic:
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, top_k, dim=1)
                topk_probs = topk_vals / topk_vals.sum(dim=1, keepdim=True)
                sampled = torch.multinomial(topk_probs, 1).squeeze(1)
                input_idx = topk_idx[torch.arange(batch_size, device=device), sampled]
            else:
                input_idx = torch.multinomial(probs, 1).squeeze(1)
        else:
            input_idx = torch.argmax(probs, dim=1)

        input_t = F.one_hot(
            input_idx,
            num_classes=model.input_dim
        ).float().unsqueeze(1)

        input_idx_cpu = input_idx.detach().cpu().numpy()

        for i in range(batch_size):
            if finished[i]:
                continue

            aa = IDX_TO_AA[input_idx_cpu[i]]

            if aa == STOP_TOKEN or aa == "Z":
                finished[i] = True
                continue

            if aa not in [PAD_TOKEN, START_TOKEN]:
                seqs[i] += aa

        if finished.all():
            break

    return seqs


# -------------------- Prototype-Guided Generation --------------------

def generate_for_one_prototype(
    model,
    proto_seq,
    target_func,
    target_length,
    n_samples,
    beta,
    device,
    prior_cache,
    min_len=5,
    max_len=35,
    perturb_std=0.01,
    temperature=1.0,
    top_k=0,
    max_attempt_rounds=10
):
    """
    Generate unique prototype-guided peptides for one prototype.

    If target_length is None, the prototype's own length is used.
    Duplicates are removed within this prototype.
    """

    proto_clean = clean_sequence(proto_seq)
    proto_len = len(proto_clean)

    if not (min_len <= proto_len <= max_len):
        log(
            f"⚠️ Skipping prototype of length {proto_len}. "
            f"Allowed range: {min_len}-{max_len}."
        )
        return []

    if target_length is None:
        this_target_length = proto_len
    else:
        this_target_length = target_length

    if not (min_len <= this_target_length <= max_len):
        log(
            f"⚠️ Skipping prototype because target length {this_target_length} "
            f"is outside allowed range {min_len}-{max_len}."
        )
        return []

    target_len_idx = get_length_bin_idx(this_target_length)

    x_proto = torch.tensor(
        model.one_hot_encode(proto_clean),
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        mu_z, _ = model.q_z(x_proto)

    proto_onehot = torch.tensor(
        model.one_hot_encode(proto_clean),
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    unique_generated = set()
    rows = []

    attempt_round = 0

    while len(unique_generated) < n_samples and attempt_round < max_attempt_rounds:
        attempt_round += 1

        remaining = n_samples - len(unique_generated)

        # Generate a little extra after first round to compensate for duplicates
        if attempt_round == 1:
            batch_size = remaining
        else:
            batch_size = max(remaining * 2, 4)

        with torch.no_grad():
            z_batch, w_batch, v_batch = sample_latents(
                model=model,
                mu_z=mu_z,
                n_samples=batch_size,
                target_func=target_func,
                target_len_idx=target_len_idx,
                device=device,
                perturb_std=perturb_std,
                prior_cache=prior_cache,
                stochastic=True
            )

            generated_seqs = decode_sequence_batch(
                model=model,
                z=z_batch,
                w=w_batch,
                v=v_batch,
                target_len=this_target_length,
                proto_onehot=proto_onehot,
                beta=beta,
                min_len=min_len,
                max_len=max_len,
                temperature=temperature,
                device=device,
                stochastic=True,
                top_k=top_k
            )

        for seq in generated_seqs:
            s_clean = clean_sequence(seq)

            if len(s_clean) < min_len:
                continue

            if s_clean in unique_generated:
                continue

            unique_generated.add(s_clean)

            rows.append({
                "Prototype_sequence": proto_clean,
                "Generated_sequence": s_clean,
                "Target_Function": target_func,
                "Target_Length": this_target_length
            })

            if len(unique_generated) >= n_samples:
                break

    if len(unique_generated) < n_samples:
        log(
            f"⚠️ Prototype length {proto_len}: generated only "
            f"{len(unique_generated)}/{n_samples} unique peptides "
            f"after {max_attempt_rounds} attempt rounds."
        )

    return rows


def generate(
    model,
    sequences,
    target_func,
    target_length,
    n_samples,
    beta,
    output_path,
    seed,
    device
):
    prior_cache = build_prior_cache(model, device)
    all_rows = []

    for proto_seq in tqdm(sequences, desc="Generating prototype-guided peptides"):
        rows = generate_for_one_prototype(
            model=model,
            proto_seq=proto_seq,
            target_func=target_func,
            target_length=target_length,
            n_samples=n_samples,
            beta=beta,
            device=device,
            prior_cache=prior_cache,
            min_len=5,
            max_len=35,
            perturb_std=0.01,
            temperature=1.0,
            top_k=0,
            max_attempt_rounds=10
        )

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if target_length is None:
        length_tag = "proto_len"
    else:
        length_tag = f"len{target_length}"

    beta_tag = str(beta)

    output_file = (
        output_path
        / f"peptides_func{target_func}_{length_tag}_beta{beta_tag}_seed{seed}_{timestamp}.tsv"
    )

    df.to_csv(output_file, sep="\t", index=False)

    log(f"✅ Saved {len(df)} peptides to {output_file}")


# -------------------- CLI --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate prototype-guided peptides conditioned on target function and optional target length"
    )

    parser.add_argument(
        "-i",
        "--input_fasta",
        required=True,
        help="Input FASTA file containing prototype sequences"
    )

    parser.add_argument(
        "-f",
        "--target_function",
        required=True,
        type=int,
        help="Target function ID"
    )

    parser.add_argument(
        "-l",
        "--target_length",
        type=int,
        default=None,
        help="Target peptide length. If omitted, each prototype uses its own length."
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=5,
        help="Number of unique analogues per prototype. Default: 5."
    )

    parser.add_argument(
        "-b",
        "--beta",
        type=float,
        default=0.65,
        help="Prototype-bias strength. Default: 0.65."
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed. If omitted, a time-based seed is used."
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="generated_peptides_prototype",
        help="Output directory"
    )

    args = parser.parse_args()

    if not (0.0 <= args.beta < 1.0):
        raise ValueError(
            f"beta must be >= 0.0 and < 1.0. Received beta={args.beta}."
        )

    if args.target_length is not None and not (5 <= args.target_length <= 35):
        raise ValueError(
            f"target_length must be between 5 and 35. Received {args.target_length}."
        )

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % 100000
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    dir_path = Path(__file__).resolve().parent

    checkpoint_path = (
        dir_path
        / "models"
        / "generative_model"
        / "PLUM_new_analysis_renew_part4_v2_004.pth"
    )

    model = load_model(checkpoint_path, device)

    sequences = [
        str(rec.seq).upper()
        for rec in SeqIO.parse(args.input_fasta, "fasta")
    ]

    if not sequences:
        raise ValueError(f"No sequences found in {args.input_fasta}")

    log(f"Loaded {len(sequences)} prototype sequences.")

    if args.target_length is None:
        log(
            f"Target function: {args.target_function}, "
            f"Target length: prototype length, "
            f"Beta: {args.beta}"
        )
    else:
        log(
            f"Target function: {args.target_function}, "
            f"Target length: {args.target_length}, "
            f"Beta: {args.beta}"
        )

    output_path = dir_path / args.output_dir

    generate(
        model=model,
        sequences=sequences,
        target_func=args.target_function,
        target_length=args.target_length,
        n_samples=args.n_samples,
        beta=args.beta,
        output_path=output_path,
        seed=seed,
        device=device
    )


if __name__ == "__main__":
    main()