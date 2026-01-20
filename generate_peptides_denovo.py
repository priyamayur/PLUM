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
from training_generative_model.generative_model import (
    PeptideCSVAE_LSTM, AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, START_TOKEN,
    length_to_bin, LENGTH_BINS, NUM_LENGTH_BINS
)

# -----------------------------
# Paths & device
# -----------------------------
dir_path = "/work/idoerg/priyanka/software/PLUM"
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

    model.eval()
    device = torch.device(device)
    model.to(device)

    # Convert target length to bin
    target_bin_idx = int(length_to_bin(target_length))  # convert to int

    # Conditioning tensors
    y_func = torch.tensor([[target_function]], dtype=torch.float32, device=device)
    y_len = F.one_hot(
        torch.tensor([target_bin_idx], dtype=torch.long),
        NUM_LENGTH_BINS
    ).float().to(device)

    # Latent variables
    z = torch.randn(1, model.z_dim, device=device)
    mu_w, logvar_w = model.p_w_prior(y_func)
    w = model.reparameterize(mu_w, logvar_w).unsqueeze(0)  # ensure 2D
    mu_v, logvar_v = model.p_v_prior(y_len)
    v = model.reparameterize(mu_v, logvar_v).unsqueeze(0)  # ensure 2D

    # Repeat for batch
    w_batch = w.repeat(n_samples_per_condition, 1)
    v_batch = v.repeat(n_samples_per_condition, 1)
    z_batch = z.repeat(n_samples_per_condition, 1)

    # --- Init LSTM state ---
    def init_lstm_state(batch_size):
        h = torch.zeros(model.lstm_layers, batch_size, model.hidden_dim, device=device)
        c = torch.zeros(model.lstm_layers, batch_size, model.hidden_dim, device=device)
        return h, c

    start_idx = torch.full((1,), AA_TO_IDX[START_TOKEN], dtype=torch.long, device=device)
    input_t_base = F.one_hot(start_idx, num_classes=model.input_dim).float().unsqueeze(1)

    # Prepare batches
    seqs = [''] * n_samples_per_condition
    finished = [False] * n_samples_per_condition

    h, c = init_lstm_state(n_samples_per_condition)
    input_t = input_t_base.repeat(n_samples_per_condition, 1, 1)
    z_batch = z.repeat(n_samples_per_condition, 1)
    w_batch = w.repeat(n_samples_per_condition, 1)
    v_batch = v.repeat(n_samples_per_condition, 1)

    # Generate sequences step by step
    for step in range(target_length):

        lstm_input = torch.cat([input_t,
                                z_batch.unsqueeze(1),
                                w_batch.unsqueeze(1),
                                v_batch.unsqueeze(1)
                                ], dim=2)

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

        # Prepare next input
        input_t = F.one_hot(input_idx, num_classes=model.input_dim).float().unsqueeze(1)

        # Append decoded AA
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

    # Ensure sequences meet min_len
    return [s for s in seqs if len(s) >= min_len]


target_function = 1  # e.g., antimicrobial
target_length = 12        # e.g., length bin index
n_samples = 10

max_len = 35
peptides = generate_peptides_denovo(model, target_function, target_length, n_samples_per_condition=n_samples,
                                  min_len=5, max_len=max_len, temperature=1,top_k=0,   device=device)

# Prepare DataFrame
df = pd.DataFrame({
    "ID": [f"peptide_{i+1}" for i in range(len(peptides))],
    "Peptide": peptides,
    "Function": [target_function] * len(peptides),
    "Length": [len(p) for p in peptides]
})

# Save CSV
output_file = os.path.join(OUTPUT_DIR, f"peptides_func{target_function}_len{target_length}.csv")
df.to_csv(output_file, index=False)
print(f"✅ Saved {len(peptides)} peptides to {output_file}")