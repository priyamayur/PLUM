# run_lenfunc_pipeline_v9_trigram_zprior.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from generative_model import (
    PeptideCSVAE_LSTM, PeptideDataset_LSTM,
    AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, START_TOKEN,
    LENGTH_BINS, NUM_LENGTH_BINS, BIN_MEANS, length_to_bin
)

# -----------------------------
# Paths & Device
# -----------------------------
dir_path = "/work/idoerg/priyanka/plum/generator/"
CSV_PATH = dir_path + "train_test/train.csv"
MODEL_DIR = dir_path + "models_lstm"
OUTPUT_DIR = dir_path + "generated_peptides"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(CSV_PATH)
sequences = df['sequence'].tolist()
functions = df['mic_class_binary'].tolist()

# Add START + STOP + PAD tokens
max_len = max(len(seq) for seq in sequences)
processed_sequences = []
for seq in sequences:
    pad_len = max_len - (len(seq) + 2)
    if pad_len < 0:
        seq = seq[:max_len-2]
    processed_sequences.append(START_TOKEN + seq + 'Z' + PAD_TOKEN*max(0,pad_len))
sequences = processed_sequences
print(f"Loaded {len(sequences)} sequences | Max length: {max_len}")




# -----------------------------
# Training function
# -----------------------------

def train_csvae_lstm(
    model, optimizer, dataset, loader, sequences, functions, max_len=35,
    z_dim=32, w_dim=4, v_dim=4, hidden_dim=128,
    batch_size=64, epochs=300, lr=1e-3,
    length_loss_weight=0.5, func_loss_weight=1.0,
    z_rec_weight=1.0, kl_z_weight=0.01, adv_weight=0.1, 
    device='cpu'
):
    device = torch.device(device)

    bin_means = torch.tensor(BIN_MEANS, device=device, dtype=torch.float32)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for x, y_func, y_len in loader:
            x, y_func, y_len = x.to(device), y_func.to(device), y_len.to(device)
            optimizer.zero_grad()
            outputs = model.forward(x, y_func, y_len, teacher_forcing=True)

            # -----------------
            # Reconstruction loss (full decoder)
            # -----------------
            x_targets = x.argmax(dim=2)
            mask = (x_targets != AA_TO_IDX[PAD_TOKEN]) & \
                   (x_targets != AA_TO_IDX[START_TOKEN]) & \
                   (x_targets != AA_TO_IDX['Z'])
            rec_loss = F.cross_entropy(
                outputs["x_logits"].view(-1, model.input_dim)[mask.view(-1)],
                x_targets.view(-1)[mask.view(-1)]
            )

            # -----------------
            # z-only reconstruction using separate decoder
            # -----------------
            x_logits_zonly = outputs["x_logits_zonly"]
            rec_loss_z = F.cross_entropy(
                x_logits_zonly.view(-1, model.input_dim)[mask.view(-1)],
                x_targets.view(-1)[mask.view(-1)]
            )

            # -----------------
            # KL divergences
            # -----------------

            kl_z = -0.5 * torch.sum(1 + outputs["logvar_z"] - outputs["mu_z"].pow(2) - outputs["logvar_z"].exp()) / x.size(0)
            kl_w = model.kl_divergence(outputs["mu_w_enc"], outputs["logvar_w_enc"],
                                       outputs["mu_w_prior"], outputs["logvar_w_prior"])
            kl_v = model.kl_divergence(outputs["mu_v_enc"], outputs["logvar_v_enc"],
                                       outputs["mu_v_prior"], outputs["logvar_v_prior"])

            # -----------------
            # Length loss from v
            # -----------------
            true_len = mask.sum(dim=1).float()
            len_logits = model.v_to_len(outputs["v"])
            len_probs = F.softmax(len_logits, dim=1)
            pred_len = torch.matmul(len_probs, bin_means)
            length_mse = F.mse_loss(pred_len, true_len)
            len_bin_idx = torch.argmax(y_len, dim=1)
            len_cls_loss = F.cross_entropy(len_logits, len_bin_idx)
            length_loss = length_loss_weight * (len_cls_loss + length_mse)

            # -----------------
            # Function classification from w
            # -----------------
            func_cls_loss = F.binary_cross_entropy(outputs["y_pred_func"], y_func) * func_loss_weight

            # -----------------
            # Adversarial term: make z uninformative about func
            # -----------------

            y_logits_from_z = model.decoder_z_to_func(outputs["z"])
            adv_entropy = F.binary_cross_entropy_with_logits(y_logits_from_z, 0.5*torch.ones_like(y_logits_from_z))
            pred_len_from_z = model.decoder_z_to_len(outputs["z"]) 
            adv_len_loss = F.mse_loss(pred_len_from_z, 0.5*torch.ones_like(pred_len_from_z))

            # Total loss
            # -----------------
            loss = rec_loss + 0.1*(kl_z) +0.1*kl_w +0.1*kl_v + length_loss + func_cls_loss + (z_rec_weight*rec_loss_z) + (adv_entropy*adv_weight) + (adv_len_loss*adv_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Rec: {rec_loss.item():.4f} | ZRec: {rec_loss_z.item():.4f} | "
              f"KLz: {kl_z.item():.4f} | KLw: {kl_w:.4f} | KLv: {kl_v:.4f} | "
              f"LenLoss: {length_loss.item():.4f} | FuncLoss: {func_cls_loss.item():.4f} | AdvEnt: {adv_entropy.item():.4f} ")

    return model

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    length_loss_weight = 1
    func_loss_weight = 1
    z_rec_weight = 1
    adv_weight = 1
    kl_z_weight = 0.001
    num_epochs  = 500
    lr=1e-3
    z_dim = 4
    w_dim = 4
    v_dim = 4
    batch_size = 64
    hidden_dim = 128
    cond_dim = 1
    max_len = 35
    device = torch.device(device)
    dataset = PeptideDataset_LSTM(sequences, functions, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_initial = PeptideCSVAE_LSTM(seq_len=max_len, z_dim=z_dim, w_dim=w_dim, v_dim=v_dim,
                              hidden_dim=hidden_dim, cond_dim=cond_dim).to(device)
    optimizer = torch.optim.Adam(model_initial.parameters(), lr=lr)
    print("Training CS-VAE LSTM...")
    model = train_csvae_lstm(model=model_initial,optimizer=optimizer,dataset=dataset,loader=loader,sequences=sequences, functions=functions, max_len=max_len,
                                z_dim=z_dim, w_dim=w_dim, v_dim=v_dim,
                                hidden_dim=hidden_dim, batch_size=batch_size,
                                epochs=num_epochs , lr=lr,
                                length_loss_weight=length_loss_weight, func_loss_weight=func_loss_weight,
                                z_rec_weight=z_rec_weight, kl_z_weight=kl_z_weight, adv_weight=adv_weight,
                                device=device)
    
    ## saving the model checkpoint and loading it to train further
    
    checkpoint = {
    "epoch": num_epochs ,   # or epochs if saving at the end
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "model_config": {
        "seq_len": max_len,
        "z_dim": z_dim,
        "w_dim": w_dim,
        "v_dim": v_dim,
        "hidden_dim": hidden_dim,
        "cond_dim": cond_dim
    },
    "training_config": {
        "length_loss_weight": length_loss_weight,
        "func_loss_weight": func_loss_weight,
        "z_rec_weight": z_rec_weight,
        "adv_weight": adv_weight,
        "lr": lr,
        "batch_size": batch_size
    }
    }

    torch.save(
        checkpoint,
        os.path.join(MODEL_DIR, "PLUM_checkpoint_v1.pth")
    )
    print(f"✅ Model checkpoint saved to {os.path.join(MODEL_DIR, 'PLUM_checkpoint_v1.pth')}")