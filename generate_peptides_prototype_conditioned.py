import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from training_generative_model.generative_model import (
    PeptideCSVAE_LSTM, AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, START_TOKEN,
    length_to_bin, LENGTH_BINS, NUM_LENGTH_BINS
)

# -----------------------------
# Paths & device
# -----------------------------
dir_path = "/mnt/c/Users/pb11/Documents/PhD/software/PLUM"
CHECKPOINT_PATH = dir_path + "/trained_models/generative_model/PLUM_checkpoint_v5.pth" 
OUTPUT_DIR = dir_path + "/generated_peptides_prototype/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 35

# -----------------------------
# Model loader
# -----------------------------
def load_model(checkpoint_path, device):
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

# -----------------------------
# Sequence utilities
# -----------------------------
def clean_sequence(seq, pad_token=PAD_TOKEN, start_token=START_TOKEN):
    """Remove padding, start token, and stop token Z from sequence."""
    return seq.replace(pad_token, '').replace('Z', '').replace(start_token, '')

# -----------------------------
# Latent sampling
# -----------------------------
def sample_latents(model, mu_z, n_samples, target_func, target_len_idx, device,
                   perturb_std=0.5, prior_cache=None, stochastic=True):
    z_proto = mu_z.repeat(n_samples, 1)
    if stochastic:
        z = z_proto + torch.randn_like(z_proto) * perturb_std
    else:
        z = z_proto + torch.randn_like(z_proto) * perturb_std

    mu_w, logvar_w = prior_cache[('w', target_func)]
    w = model.reparameterize(mu_w, logvar_w).repeat(n_samples, 1) if stochastic else mu_w.repeat(n_samples, 1)

    mu_v, logvar_v = prior_cache[('v', target_len_idx)]
    v = model.reparameterize(mu_v, logvar_v).repeat(n_samples, 1) if stochastic else mu_v.repeat(n_samples, 1)

    return z, w, v

# -----------------------------
# Sequence decoding
# -----------------------------
def decode_sequence_batch(model, z, w, v, target_len, proto_onehot=None, beta=0.0,
                          min_len=5, max_len=35, temperature=0.8, device='cpu',
                          stochastic=True, top_k=0):
    batch_size = z.size(0)
    seqs = [''] * batch_size
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    h = torch.zeros(model.lstm_layers, batch_size, model.hidden_dim, device=device)
    c = torch.zeros(model.lstm_layers, batch_size, model.hidden_dim, device=device)

    start_idx_tensor = torch.full((batch_size,), AA_TO_IDX[START_TOKEN], device=device, dtype=torch.long)
    input_t = F.one_hot(start_idx_tensor, num_classes=model.input_dim).float().unsqueeze(1)

    Z_idx = AA_TO_IDX['Z']
    PAD_idx = AA_TO_IDX[PAD_TOKEN]

    for t in range(target_len):
        lstm_input = torch.cat([input_t, z.unsqueeze(1), w.unsqueeze(1), v.unsqueeze(1)], dim=2)
        out, (h, c) = model.decoder_lstm(lstm_input, (h, c))
        logits = model.out_x(out).squeeze(1) / max(temperature, 1e-6)

        logits[finished, :] = -1e9
        logits[finished, PAD_idx] = 0

        if t < min_len:
            logits[:, Z_idx] = -1e9
        if t == target_len - 1:
            logits[:, :] = -1e9
            logits[:, Z_idx] = 0

        probs = F.softmax(logits, dim=1)
        if proto_onehot is not None and t < proto_onehot.size(1) and beta > 0:
            probs = beta * proto_onehot[:, t, :] + (1 - beta) * probs

        if stochastic:
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(probs, top_k, dim=1)
                topk_probs = topk_vals / topk_vals.sum(dim=1, keepdim=True)
                sampled = torch.multinomial(topk_probs, 1).squeeze(1)
                input_idx = topk_idx[torch.arange(probs.size(0)), sampled]
            else:
                input_idx = torch.multinomial(probs, 1).squeeze(1)
        else:
            input_idx = torch.argmax(probs, dim=1)

        input_t = F.one_hot(input_idx, num_classes=model.input_dim).float().unsqueeze(1)

        for i, idx in enumerate(input_idx):
            if finished[i]: continue
            aa = IDX_TO_AA[idx.item()]
            if aa == 'Z': finished[i] = True
            if aa not in [PAD_TOKEN, START_TOKEN]: seqs[i] += aa

        if finished.all(): break

    return seqs

# -----------------------------
# Prior caching
# -----------------------------
def cache_priors(model, target_func, target_len_bin, device):
    y_func_tensor = torch.tensor([[target_func]], dtype=torch.float32, device=device)
    mu_w, logvar_w = model.p_w_prior(y_func_tensor)

    y_len_tensor = F.one_hot(torch.tensor([target_len_bin], device=device), NUM_LENGTH_BINS).float()
    mu_v, logvar_v = model.p_v_prior(y_len_tensor)

    return {('w', target_func):(mu_w, logvar_w), ('v', target_len_bin):(mu_v, logvar_v)}

# -----------------------------
# Main generation function
# -----------------------------
def generate_prototype_analogues(model, sequences, target_func, target_length,
                                 n_analogues_per_sequence=25, perturb_std=0.5,
                                 min_len=5, max_len=35, temperature=0.8, beta=0.0,
                                 device='cpu', stochastic=True, top_k=0):
    model.eval()
    device = torch.device(device)
    model.to(device)

    target_len_bin = int(np.argmax(length_to_bin(target_length)))
    decode_len = target_length
    prior_cache = cache_priors(model, target_func, target_len_bin, device)

    all_results = []

    with torch.no_grad():
        for proto_seq in tqdm(sequences, desc="Generating prototypes", unit="prototype"):
            proto_clean = clean_sequence(proto_seq)

            x_proto = torch.tensor(model.one_hot_encode(proto_seq), dtype=torch.float32).unsqueeze(0).to(device)
            mu_z, logvar_z = model.q_z(x_proto)

            proto_onehot = torch.tensor(model.one_hot_encode(proto_seq), dtype=torch.float32, device=device).unsqueeze(0)
            z_batch, w_batch, v_batch = sample_latents(model, mu_z, n_analogues_per_sequence,
                                                       target_func, target_len_bin,
                                                       device, perturb_std, prior_cache, stochastic)

            seqs = decode_sequence_batch(model, z_batch, w_batch, v_batch, decode_len,
                                         proto_onehot, beta, min_len, max_len,
                                         temperature, device, stochastic, top_k)

            for s in seqs:
                s_clean = clean_sequence(s)
                if len(s_clean) >= min_len:
                    all_results.append({
                        "prototype_sequence": proto_clean,
                        "generated_sequence": s_clean,
                        "target_func": target_func,
                        "target_length": target_length,
                        "prototype_length": len(proto_clean),
                        "generated_length": len(s_clean)
                    })

    return pd.DataFrame(all_results)

# -----------------------------
# Run example
# -----------------------------
model = load_model(CHECKPOINT_PATH, device)

sequences = [
    "MKWVTFISLLFLFSSAYSR",
    "GIGKFLHSAKKFGKAFV",
    "PEDPQRRYQEEQRRE",
    "RNSVRNRVMLWRTKR"
]
target_func = 0
target_length = 13

df = generate_prototype_analogues(model, sequences, target_func, target_length,
                                  n_analogues_per_sequence=5, beta=0.45,
                                  temperature=1, stochastic=True, device=device)

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, f"prototype_analogues_func{target_func}_len{target_length}.csv")
df.to_csv(output_file, index=False)
print(f"✅ Saved {len(df)} analogues to {output_file}")
