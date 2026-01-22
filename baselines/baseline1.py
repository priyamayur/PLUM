# baseline_cvae_updated.py
import os
import csv
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# -----------------------------
# Amino acids and dataset
# -----------------------------
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------
# Dataset
# -----------------------------
class PeptideDataset:
    def __init__(self, peptides, y_f, y_s=None):
        self.peptides = peptides
        self.y_f = np.array(y_f, dtype=np.int64)
        self.y_s = np.array([len(p) for p in peptides], dtype=np.int64) if y_s is None else np.array(y_s)
        self.max_len = max(len(p) for p in peptides)

    def create_targets_with_mask(self):
        N = len(self.peptides)
        X = np.zeros((N, self.max_len, len(AMINO_ACIDS)), dtype=np.float32)
        targets = np.full((N, self.max_len), fill_value=-100, dtype=np.int64)

        for i, p in enumerate(self.peptides):
            for j, aa in enumerate(p):
                X[i, j, AA_TO_IDX[aa]] = 1.0
                targets[i, j] = AA_TO_IDX[aa]

        return X, targets, self.y_s, self.y_f

    def get_batch(self, batch_size=None, shuffle=False):
        X, targets, lengths, yf = self.create_targets_with_mask()
        if shuffle:
            perm = np.random.permutation(len(X))
            X, targets, lengths, yf = X[perm], targets[perm], lengths[perm], yf[perm]
        if batch_size is None:
            return X, targets, lengths, yf
        else:
            return X[:batch_size], targets[:batch_size], lengths[:batch_size], yf[:batch_size]

# -----------------------------
# CVAE Models
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, latent_dim=32, seq_len=25, cond_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * seq_len + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, cond):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(torch.cat([x, cond], dim=1)))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=128, output_dim=20, seq_len=25, cond_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.fc1 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, seq_len * output_dim)
        self.output_dim = output_dim

    def forward(self, z, cond):
        h = F.relu(self.fc1(torch.cat([z, cond], dim=1)))
        h = F.relu(self.fc2(h))
        x_recon = self.fc_out(h).view(-1, self.seq_len, self.output_dim)
        return x_recon

# -----------------------------
# Utilities
# -----------------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

# -----------------------------
# Training
# -----------------------------
def train_cvae(csv_path, epochs=50, batch_size=16, lr=1e-3, latent_dim=32, model_dir="models", device=None):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    peptides = df['sequence'].tolist()
    y_f = df['mic_class_binary'].astype(int).tolist()

    dataset = PeptideDataset(peptides, y_f)
    seq_len = dataset.max_len

    X_np, targets_np, lengths_np, yf_np = dataset.create_targets_with_mask()
    X = torch.tensor(X_np, dtype=torch.float32)
    targets = torch.tensor(targets_np, dtype=torch.long)
    cond = torch.tensor(np.stack([yf_np, lengths_np / seq_len], axis=1), dtype=torch.float32)

    data_loader = DataLoader(TensorDataset(X, targets, cond), batch_size=batch_size, shuffle=True)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    encoder = Encoder(input_dim=20, hidden_dim=128, latent_dim=latent_dim, seq_len=seq_len, cond_dim=2).to(device)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=128, output_dim=20, seq_len=seq_len, cond_dim=2).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        total_loss = 0.0
        for x_batch, target_batch, cond_batch in data_loader:
            x_batch, target_batch, cond_batch = x_batch.to(device), target_batch.to(device), cond_batch.to(device)
            optimizer.zero_grad()
            mu, logvar = encoder(x_batch, cond_batch)
            z = reparameterize(mu, logvar)
            x_recon_logits = decoder(z, cond_batch)
            loss_rec = ce_loss(x_recon_logits.view(-1, 20), target_batch.view(-1))
            loss_kl = kl_divergence(mu, logvar)
            loss = loss_rec + 0.01 * loss_kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(X)
        print(f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.4f} (rec: {loss_rec.item():.4f}, kl: {loss_kl.item():.4f})")

    # Save models
    torch.save(encoder.state_dict(), os.path.join(model_dir, "cvae_encoder.pth"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "cvae_decoder.pth"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({"seq_len": seq_len, "latent_dim": latent_dim}, f)

    return encoder, decoder, seq_len, latent_dim, dataset

# -----------------------------
# Peptide generation
# -----------------------------
def generate_peptide(decoder, latent_dim, target_function=1, target_length=10, temperature=1.0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device).eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        cond = torch.tensor([[target_function, target_length / decoder.seq_len]], dtype=torch.float32).to(device)
        logits = decoder(z, cond)[:, :target_length, :]
        probs = F.softmax(logits / temperature, dim=-1)
        peptide = "".join([AMINO_ACIDS[torch.multinomial(probs[0, i], 1).item()] for i in range(target_length)])
    return peptide


def generate_lenwin_analogues(
    prototypes,
    y_f,
    encoder,
    decoder,
    latent_dim,
    csv_path,
    fasta_path,
    device=None,
    n=5,
    window=5,
    min_len=5,
    max_len=35,
    temperature=1.0,
    perturb_std=0.05,
    alpha=0.8,
    mode="multinomial",
    top_k=None
):
    """
    Generate analogues with target length = prototype_length ± window
    while respecting hard bounds [min_len, max_len].
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    decoder.to(device).eval()

    seq_len = decoder.seq_len
    rows = []
    fasta = []
    print("now generating")
    for pid, (proto_seq, proto_func) in enumerate(zip(prototypes, y_f)):
        print("current pid", str(pid), "out of total=", len(prototypes))
        proto_len = len(proto_seq)

        # Length window with hard bounds
        tgt_lengths = range(
            max(min_len, proto_len - window),
            min(max_len, proto_len + window) + 1
        )

        # One-hot encode prototype
        encoded = torch.zeros((1, seq_len, len(AMINO_ACIDS)), device=device)
        for i, aa in enumerate(proto_seq):
            encoded[0, i, AA_TO_IDX[aa]] = 1.0

        cond_proto = torch.tensor(
            [[proto_func, proto_len / seq_len]],
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            mu, _ = encoder(encoded, cond_proto)
            z_proto = mu.clone()

        total_loop = len(tgt_lengths) * n * 2
        print("total loop", str(total_loop))
        total_loop_count = 0
        for tgt_func in [0, 1]:
            for tgt_len in tgt_lengths:
                tgt_cond = torch.tensor(
                    [[tgt_func, tgt_len / seq_len]],
                    dtype=torch.float32,
                    device=device
                )

                for _ in range(n):
                    total_loop_count += 1
                    # print("current loop", str(total_loop_count), "out of total=", str(total_loop))

                    # Perturb z_proto
                    noise = torch.randn_like(z_proto) * perturb_std
                    z = alpha * z_proto + (1 - alpha) * noise

                    with torch.no_grad():
                        logits = decoder(z, tgt_cond)[:, :tgt_len, :]
                        probs = F.softmax(logits / temperature, dim=-1)

                    gen = []
                    for i in range(tgt_len):
                        if mode == "argmax":
                            aa_idx = torch.argmax(probs[0, i]).item()
                        elif mode == "multinomial":
                            prob = probs[0, i]
                            if top_k is not None and top_k < len(prob):
                                top_probs, top_idx = torch.topk(prob, top_k)
                                top_probs = top_probs / top_probs.sum()
                                aa_idx = top_idx[torch.multinomial(top_probs, 1).item()].item()
                            else:
                                aa_idx = torch.multinomial(prob, 1).item()
                        else:
                            raise ValueError("mode must be 'multinomial' or 'argmax'")
                        gen.append(AMINO_ACIDS[aa_idx])

                    gen_seq = "".join(gen)

                    rows.append({
                        "prototype_sequence": proto_seq,
                        "generated_sequence": gen_seq,
                        "original_func": proto_func,
                        "target_func": tgt_func,
                        "prototype_length": proto_len,
                        "target_length": tgt_len,
                        "length_delta": tgt_len - proto_len
                    })

                    fasta.append(
                        f">proto{pid}_origF{proto_func}_tgtF{tgt_func}_L{tgt_len}\n{gen_seq}"
                    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(fasta_path, "w") as f:
        f.write("\n".join(fasta))

    print(
        f"✅ Length-window CVAE analogues saved to:\n"
        f"  CSV   → {csv_path}\n"
        f"  FASTA → {fasta_path}"
    )
# -----------------------------
# Main workflow
# -----------------------------
def main():
    print("Starting CVAE peptide generation workflow...")
    dir_path = "/work/idoerg/priyanka/plum/generator/"
    TRAIN_CSV = dir_path + "train_test/train.csv"
    TEST_CSV = dir_path + "train_test/test.csv"

    MODEL_DIR = dir_path + "models"
    RANDOM_FASTA = dir_path + "generated_peptides/random_generation_baseline_1.fasta"
    ANALOGUE_CSV = dir_path + "generated_peptides/analogues_generation_baseline_1.csv"
    os.makedirs(os.path.dirname(RANDOM_FASTA), exist_ok=True)

    # -----------------------------
    # Train CVAE
    # -----------------------------
    print("Training CVAE model...")
    encoder, decoder, seq_len, latent_dim, train_dataset = train_cvae(
        TRAIN_CSV, epochs=500, latent_dim=8, model_dir=MODEL_DIR
    )
    print("Training completed.")
    # -----------------------------
    # Save models and metadata
    # -----------------------------
    model_dir = MODEL_DIR + "/baseline_1"
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1️⃣ Save full models (architecture + weights)
    torch.save(encoder, os.path.join(model_dir, "cvae_encoder_full.pth"))
    torch.save(decoder, os.path.join(model_dir, "cvae_decoder_full.pth"))

    # 2️⃣ Save state_dicts as backup (optional)
    torch.save(encoder.state_dict(), os.path.join(model_dir, "cvae_encoder_state.pth"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "cvae_decoder_state.pth"))

    # 3️⃣ Save config / metadata
    config = {
        "seq_len": seq_len,
        "latent_dim": latent_dim,
        "amino_acids": list(AMINO_ACIDS)
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)

    print(f"✅ Models and metadata saved in '{model_dir}'")

    # -----------------------------
    # Random peptide generation
    # -----------------------------
    print("Generating random peptides...")
    with open(RANDOM_FASTA, "w") as f:
        count = 1
        for func_label in [0, 1]:
            for length in range(5, seq_len + 1):
                for _ in range(330):
                    pep = generate_peptide(decoder, latent_dim, target_function=func_label,
                                           target_length=length, temperature=1.0)
                    f.write(f">peptide_{count}_func{func_label}_len{length}\n{pep}\n")
                    count += 1
    print(f"✅ Random peptides saved to: {RANDOM_FASTA}")

    # -----------------------------
    # Generate analogues for test set
    # -----------------------------
    print("Generating multi-condition analogues from test prototypes...")
    df_test = pd.read_csv(TEST_CSV)
    test_dataset = PeptideDataset(df_test['sequence'].tolist(), df_test['mic_class_binary'].tolist())
    
        # -----------------------------
    # Length-window analogue generation (CVAE)
    # -----------------------------
    print("Generating length-window CVAE analogues...")

    CSV_LENWIN = dir_path + "generated_peptides/analogues_lenwin_baseline_1_2.csv"
    FASTA_LENWIN = dir_path + "generated_peptides/analogues_lenwin_baseline_1.fasta"

    df_test = pd.read_csv(TEST_CSV)
    test_dataset = PeptideDataset(df_test['sequence'].tolist(), df_test['mic_class_binary'].tolist())

    generate_lenwin_analogues(
        prototypes=test_dataset.peptides,
        y_f=test_dataset.y_f,
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        csv_path=CSV_LENWIN,
        fasta_path=FASTA_LENWIN,
        device=device,
        n=100,               
        window=7,         
        min_len=5,
        max_len=35,
        temperature=1.0,
        perturb_std=0.01,
        alpha=0.4,
        mode="multinomial",
        top_k=None
    )
if __name__ == "__main__":
    main()
