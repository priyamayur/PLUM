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
    AA_TO_IDX, IDX_TO_AA, PAD_TOKEN, START_TOKEN, STOP_TOKEN,
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
max_len = max(len(seq) for seq in sequences) + 2  # +2 for START and STOP
processed_sequences = []
for seq in sequences:
    pad_len = max_len - (len(seq) + 2)
    processed_sequences.append(START_TOKEN + seq + STOP_TOKEN + PAD_TOKEN*max(0,pad_len))
sequences = processed_sequences


print(f"Loaded {len(sequences)} sequences | Max length: {max_len}")

import math

def kl_z_schedule(epoch, min_w=0.001, max_w=0.1, center=150, scale=40):
    w = max_w / (1 + math.exp(-(epoch - center) / scale))
    return max(min_w, min(w, max_w))


# -----------------------------
# Training function
# -----------------------------

def train_csvae_lstm(
    model, optimizer, dataset, loader, sequences, functions, max_len=35,
    z_dim=32, w_dim=4, v_dim=4, hidden_dim=128,
    batch_size=64, epochs=300, lr=1e-3,
    length_loss_weight=0.5, func_loss_weight=1.0,
    z_rec_weight=1.0, kl_z_weight=0.1, kl_w_weight=0.1, kl_v_weight=0.1, adv_weight=1.0,
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
                   (x_targets != AA_TO_IDX[STOP_TOKEN])
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
            
            
            kl_per_dim = -0.5 * (1 + outputs["logvar_z"] - outputs["mu_z"].pow(2) - outputs["logvar_z"].exp())
            min_info_threshold = 0.08
            under_info_penalty = torch.clamp(min_info_threshold - kl_per_dim, min=0.0)
            
            kl_z = kl_per_dim.mean() + under_info_penalty.mean()
            
            kl_w = model.kl_divergence(outputs["mu_w_enc"], outputs["logvar_w_enc"],
                                       outputs["mu_w_prior"], outputs["logvar_w_prior"])
            kl_v = model.kl_divergence(outputs["mu_v_enc"], outputs["logvar_v_enc"],
                                       outputs["mu_v_prior"], outputs["logvar_v_prior"])

            # -----------------
            # Length loss
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
            # Function loss from w NEW
            # -----------------

            w_latent = outputs["w"]

            if model.training:
                # Add Gaussian noise (std=0.1) to 'harden' the representation
                noise = torch.randn_like(w_latent) * 0.1
                w_for_cls = w_latent + noise
            else:
                w_for_cls = w_latent

            # Get logits from the updated MLP
            y_logits_func = model.w_to_func(w_for_cls)


            # Switch to BCEWithLogitsLoss
            func_cls_loss = F.binary_cross_entropy_with_logits(
                y_logits_func,
                y_func,
            ) * func_loss_weight

            # -----------------
            # Adversarial term: make z uninformative about func
            # -----------------
            
            y_logits_from_z = model.decoder_z_to_func(outputs["z"])
            adv_entropy = F.binary_cross_entropy(y_logits_from_z, 0.5*torch.ones_like(y_logits_from_z))
            

            # # 1. Get Logits (Detach from encoder to keep the adversary as a 'Fixed Expert')
            logits = model.decoder_z_to_len(outputs["z"].detach())

            # 2. Create the Uniform Target (The 'Blind' state)
            num_bins = logits.size(-1)
            uniform_target = torch.full_like(logits, 1.0 / num_bins)

            # 3. Calculate KL-Divergence
            # F.kl_div requires log-probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            adv_len_loss = F.kl_div(log_probs, uniform_target, reduction='batchmean')
            # -----------------
            # Total loss
            # -----------------
            loss = rec_loss + kl_z_weight*(kl_z) +kl_w_weight*kl_w +kl_v_weight*kl_v + length_loss + func_cls_loss +  + (adv_entropy*adv_weight) + (adv_len_loss*adv_weight) + (z_rec_weight*rec_loss_z)#+ length_loss + func_cls_loss # 0.1*(kl_z) + 0.1*kl_v + length_loss + func_cls_loss + (adv_entropy*adv_weight) + (adv_len_loss*adv_weight)#(z_rec_weight*rec_loss_z)#+ # + func_cls_loss #+ (z_rec_weight*rec_loss_z) #+ (adv_entropy*adv_weight) + (adv_len_loss*adv_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Rec: {rec_loss.item():.4f} | ZRec: {rec_loss_z.item():.4f} | "
              f"KLz: {kl_z.item():.4f} | KLw: {kl_w:.4f} | KLv: {kl_v:.4f} | "
              f"LenLoss: {length_loss.item():.4f} | FuncLoss: {func_cls_loss.item():.4f} | AdvEnt: {adv_entropy.item():.4f} | AdvLen: {adv_len_loss.item():.4f}")

    return model


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    length_loss_weight = 2
    func_loss_weight = 2
    z_rec_weight = 1
    adv_weight = 1
    kl_z_weight = 1
    kl_w_weight = 0.023
    kl_v_weight = 0.03
    num_epochs  = 150
    lr=1e-3
    z_dim = 8
    w_dim = 8
    v_dim = 8
    batch_size = 64
    hidden_dim = 128
    cond_dim = 1
    print(f"Training config | Length Loss: {length_loss_weight} | Func Loss: {func_loss_weight} | Z Rec Loss: {z_rec_weight} | KL z weight: {kl_z_weight} | KL w weight: {kl_w_weight} | KL v weight: {kl_v_weight} | Adv weight: {adv_weight} | Epochs: {num_epochs} | LR: {lr} | z_dim: {z_dim} | w_dim: {w_dim} | v_dim: {v_dim}")
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
                                z_rec_weight=z_rec_weight, kl_z_weight=kl_z_weight, kl_w_weight=kl_w_weight, kl_v_weight=kl_v_weight, adv_weight=adv_weight,
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
    version = "renew_part4_v2"
    torch.save(
        checkpoint,
        os.path.join(MODEL_DIR, f"PLUM_new_analysis_{version}.pth")
    )
    print(f"✅ Model checkpoint saved to {os.path.join(MODEL_DIR, f'PLUM_new_analysis_{version}.pth')}")