import torch
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import trange
import time
from training_generative_model.generative_model import (
    PeptideCSVAE_LSTM, AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, START_TOKEN,
    length_to_bin, LENGTH_BINS, NUM_LENGTH_BINS
)

# -----------------------------
# Paths & device
# -----------------------------
dir_path = "/mnt/c/Users/pb11/Documents/PhD/software/PLUM"
CHECKPOINT_PATH = dir_path + "/trained_models/generative_model/PLUM_checkpoint_v5.pth" #PLUM_checkpoint_v5 PLUM_checkpoint_v_test_1
OUTPUT_DIR =  dir_path + "/generated_peptides_denovo/"
TEST_CSV =  dir_path + "/data/test.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
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
print("✅ Model loaded.")

def generate_peptides_denovo(
    model,
    target_function,
    target_length,
    n_samples_per_condition=1,
    min_len=5,
    max_len=35,
    temperature=1.0,
    top_k=0,  # 0 = disabled
    device='cpu'
):
    """
    Generate peptides conditioned on a target function and exact target length.

    Returns a list of sequences of length `target_length`.
    """
    import torch.nn.functional as F
    import numpy as np
    import torch

    model.eval()
    device = torch.device(device)
    model.to(device)

    # --- Convert target length to bin (safe handling) ---
    bins = np.asarray(length_to_bin(target_length))
    if bins.size == 0:
        raise ValueError(f"No length bin found for target length {target_length}")
    target_bin_idx = int(bins[0])  # take first bin

    # --- Conditioning tensors ---
    y_func = torch.tensor([[target_function]], dtype=torch.float32, device=device)
    y_len = F.one_hot(torch.tensor([target_bin_idx], dtype=torch.long), NUM_LENGTH_BINS).float().to(device)

    # --- Sample latent variables ---
    z = torch.randn(1, model.z_dim, device=device)
    mu_w, logvar_w = model.p_w_prior(y_func)
    w = model.reparameterize(mu_w, logvar_w)
    if w.dim() == 1: 
        w = w.unsqueeze(0)
    mu_v, logvar_v = model.p_v_prior(y_len)
    v = model.reparameterize(mu_v, logvar_v)
    if v.dim() == 1:
        v = v.unsqueeze(0)

    # --- Repeat for batch ---
    z_batch = z.repeat(n_samples_per_condition, 1)
    w_batch = w.repeat(n_samples_per_condition, 1)
    v_batch = v.repeat(n_samples_per_condition, 1)

    # --- Initialize LSTM ---
    h = torch.zeros(model.lstm_layers, n_samples_per_condition, model.hidden_dim, device=device)
    c = torch.zeros(model.lstm_layers, n_samples_per_condition, model.hidden_dim, device=device)

    start_idx = torch.full((1,), AA_TO_IDX[START_TOKEN], dtype=torch.long, device=device)
    input_t = F.one_hot(start_idx, num_classes=model.input_dim).float().unsqueeze(1)
    input_t = input_t.repeat(n_samples_per_condition, 1, 1)

    # --- Prepare sequences ---
    seqs = [''] * n_samples_per_condition
    finished = [False] * n_samples_per_condition

    # --- Timer ---
    start_time = time.time()
    
    # --- Generate peptide step by step with progress bar ---
    for step in trange(target_length, desc="Generating peptides"):
        lstm_input = torch.cat([input_t, z_batch.unsqueeze(1), w_batch.unsqueeze(1), v_batch.unsqueeze(1)], dim=2)
        out, (h, c) = model.decoder_lstm(lstm_input, (h, c))
        logits = model.out_x(out).squeeze(1)  # [batch, vocab]

        # --- Temperature scaling ---
        logits = logits / max(temperature, 1e-6)

        # --- Top-k sampling ---
        if top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=1)
            probs = F.softmax(topk_vals, dim=1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            input_idx = topk_idx[torch.arange(n_samples_per_condition), sampled]
        else:
            probs = F.softmax(logits, dim=1)
            input_idx = torch.multinomial(probs, 1).squeeze(1)

        # --- Prepare next input ---
        input_t = F.one_hot(input_idx, num_classes=model.input_dim).float().unsqueeze(1)

        # --- Append amino acids ---
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
    print(f"⏱  Finished generating {n_samples_per_condition} peptides in {elapsed:.2f} seconds.")

    # --- Filter sequences shorter than min_len ---
    return [s for s in seqs if len(s) >= min_len]



target_function = 1  # e.g., antimicrobial
target_length = 12        # e.g., length bin index
n_samples = 10

max_len = 35

print(f"🧪 Generating peptides with function {target_function} and length {target_length}...")
peptides = generate_peptides_denovo(model, target_function, target_length, n_samples_per_condition=n_samples,
                                  min_len=5, max_len=max_len, temperature=1,top_k=0,   device=device)
print(f"✅ Generated {len(peptides)} peptides.")

# Prepare DataFrame
df = pd.DataFrame({
    "ID": [f"peptide_{i+1}" for i in range(len(peptides))],
    "Peptide": peptides,
    "Function": [target_function] * len(peptides),
    "Length": [len(p) for p in peptides]
})

# Save CSV
# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, f"peptides_func{target_function}_len{target_length}.csv")
df.to_csv(output_file, index=False)
print(f"✅ Saved {len(peptides)} peptides to {output_file}")