# cvae_lstm_updated.py
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Amino acids and dataset
# -----------------------------
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

# -----------------------------
# Dataset
# -----------------------------
class PeptideDataset(Dataset):
    def __init__(self, peptides, y_f, y_s=None):
        self.peptides = peptides
        self.y_f = np.array(y_f, dtype=np.int64)
        if y_s is None:
            self.y_s = np.array([len(p) for p in peptides], dtype=np.int64)
        else:
            self.y_s = np.array(y_s, dtype=np.int64)
        self.max_len = max(len(p) for p in peptides)

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, idx):
        peptide = self.peptides[idx]
        length = len(peptide)
        # One-hot encode
        x = np.zeros((self.max_len, len(AMINO_ACIDS)), dtype=np.float32)
        target = np.full((self.max_len,), fill_value=-100, dtype=np.int64)
        for i, aa in enumerate(peptide):
            x[i, AA_TO_IDX[aa]] = 1.0
            target[i] = AA_TO_IDX[aa]
        cond = np.array([self.y_f[idx], length / self.max_len], dtype=np.float32)
        return x, target, cond, length

# -----------------------------
# CVAE LSTM Models
# -----------------------------
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, latent_dim=32, cond_dim=2, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + cond_dim, latent_dim)

    def forward(self, x, cond):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        h_cond = torch.cat([h, cond], dim=1)
        mu = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        return mu, logvar

class DecoderLSTM(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=128, output_dim=20, seq_len=25, cond_dim=2, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_to_hidden = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z, cond):
        batch_size = z.size(0)
        h0 = self.latent_to_hidden(torch.cat([z, cond], dim=1)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        inputs = torch.zeros(batch_size, self.seq_len, len(AMINO_ACIDS), device=z.device)
        out, _ = self.lstm(inputs, (h0, c0))
        logits = self.fc_out(out)
        return logits

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

# -----------------------------
# Training
# -----------------------------
def train_cvae_lstm(csv_path, epochs=50, batch_size=16, lr=1e-3, latent_dim=32, model_dir="models", device=None):
    os.makedirs(model_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    peptides = df['sequence'].tolist()
    y_f = df['mic_class_binary'].astype(int).tolist()

    dataset = PeptideDataset(peptides, y_f)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    encoder = EncoderLSTM(input_dim=20, hidden_dim=128, latent_dim=latent_dim, cond_dim=2).to(device)
    decoder = DecoderLSTM(latent_dim=latent_dim, hidden_dim=128, output_dim=20,
                          seq_len=dataset.max_len, cond_dim=2).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        total_loss = 0.0
        for x_batch, target_batch, cond_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            target_batch = target_batch.to(device)
            cond_batch = cond_batch.to(device)

            optimizer.zero_grad()
            mu, logvar = encoder(x_batch, cond_batch)
            z = reparameterize(mu, logvar)
            x_recon_logits = decoder(z, cond_batch)
            loss_rec = ce_loss(x_recon_logits.view(-1, 20), target_batch.view(-1))
            loss_kl = kl_divergence(mu, logvar)
            loss = loss_rec + 0.01*loss_kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Recon Loss: {loss_rec.item():.4f} | KL Loss: {loss_kl.item():.4f}")

    # Save models
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder_lstm.pth"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "decoder_lstm.pth"))
    print("✅ Models saved.")

    return encoder, decoder, dataset

# -----------------------------
# Peptide generation
# -----------------------------
def generate_peptide_lstm(decoder, latent_dim, target_function=1, target_length=10, temperature=1.0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device).eval()
    with torch.no_grad():
        z = torch.randn(1, latent_dim).to(device)
        cond = torch.tensor([[target_function, target_length / decoder.seq_len]], dtype=torch.float32).to(device)
        logits = decoder(z, cond)[:, :target_length, :]
        probs = F.softmax(logits / temperature, dim=-1)
        peptide = "".join([AMINO_ACIDS[torch.multinomial(probs[0, i], 1).item()] for i in range(target_length)])
    return peptide

# -----------------------------
# Random peptide generation
# -----------------------------
def generate_random_peptides_lstm(decoder, latent_dim, seq_len, output_fasta, device=None,
                                  temperature=0.8, peptides_per_func=10):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device).eval()
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

    count = 1
    with open(output_fasta, "w") as f:
        for func_label in [0, 1]:
            for length in range(5, seq_len + 1):
                for _ in range(peptides_per_func):
                    pep = generate_peptide_lstm(decoder, latent_dim,
                                                target_function=func_label,
                                                target_length=length,
                                                temperature=temperature,
                                                device=device)
                    f.write(f">peptide_{count}_func{func_label}_len{length}\n{pep}\n")
                    count += 1
    print(f"✅ Random peptides saved to: {output_fasta}")
EOS_TOKEN = '<EOS>'
EOS_IDX = AA_TO_IDX[EOS_TOKEN]
def generate_peptide(decoder, z, func, length, temperature=1.0, mode='multinomial', top_k=None):
    """
    Generate a peptide sequence from a latent vector z.

    Args:
        decoder: trained DecoderLSTM
        z: latent vector (1, latent_dim)
        func: target function label (0 or 1)
        length: desired peptide length
        temperature: softmax temperature
        mode: 'multinomial' (stochastic) or 'argmax' (deterministic)
        top_k: if set (int), restrict multinomial sampling to top-k amino acids
    """
    device = z.device
    decoder.eval()

    with torch.no_grad():
        cond = torch.tensor([[func, length / decoder.max_len]], dtype=torch.float32, device=device)
        logits = decoder(z, cond, targets=None, teacher_forcing_ratio=0.0)
        logits = logits[:, :length, :]
        probs = F.softmax(logits / temperature, dim=-1)

        seq = []
        for t in range(length):
            if mode == 'argmax':
                idx = torch.argmax(probs[0, t]).item()
            elif mode == 'multinomial':
                prob = probs[0, t]
                if top_k is not None and top_k < len(prob):
                    # Top-k filtering
                    top_probs, top_idx = torch.topk(prob, top_k)
                    top_probs = top_probs / top_probs.sum()  # renormalize
                    idx = top_idx[torch.multinomial(top_probs, 1).item()].item()
                else:
                    idx = torch.multinomial(prob, 1).item()
            else:
                raise ValueError("mode must be 'multinomial' or 'argmax'")

            if idx == EOS_IDX:
                break
            if idx < len(AMINO_ACIDS):
                seq.append(AMINO_ACIDS[idx])

    return "".join(seq)
VOCAB_SIZE = len(AMINO_ACIDS)
def generate_length_window_analogues(
    peptides,
    y_f,
    encoder,
    decoder,
    latent_dim,
    csv_path,
    fasta_path,
    n=5,
    window=5,
    min_len=5,
    max_len=35,
    perturb_std=0.01,
    temperature=1.0,
    mode="multinomial",
    top_k=None
):
    """
    Generate analogues where target length is constrained to:
        [prototype_length - window, prototype_length + window]
    while respecting hard bounds [min_len, max_len].

    Args:
        peptides: list[str] – prototype sequences (unpadded)
        y_f: list[int] – function labels
        encoder, decoder: trained models
        latent_dim: latent dimension
        csv_path, fasta_path: output paths
        n: number of samples per (func, length)
        window: +/- AA window around prototype length
        min_len, max_len: hard bounds
        perturb_std: latent noise std
        temperature, mode, top_k: decoding controls
    """

    device = next(encoder.parameters()).device
    encoder.eval()
    decoder.eval()

    rows = []
    fasta_entries = []

    for pid, (seq, func) in enumerate(zip(peptides, y_f)):
        proto_len = len(seq)
        print("current pid", str(pid), "out of total=", len(peptides))
        # Determine allowed target lengths
        tgt_lengths = range(
            max(min_len, proto_len - window),
            min(max_len, proto_len + window) + 1
        )

        # One-hot encode prototype (no padding length used)
        x = torch.zeros(1, decoder.max_len, VOCAB_SIZE, device=device)
        for i, aa in enumerate(seq):
            x[0, i, AA_TO_IDX[aa]] = 1.0

        cond_proto = torch.tensor(
            [[func, proto_len / decoder.max_len]],
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():
            mu, _ = encoder(
                x,
                cond_proto,
                torch.tensor([proto_len], device=device)
            )
        total_loop = len(tgt_lengths) * n * 2
        print("total loop", str(total_loop))
        total_loop_count = 0
        for tgt_func in [0, 1]:
            for tgt_len in tgt_lengths:
                for _ in range(n):
                    total_loop_count += 1
                    # print("current loop", str(total_loop_count), "out of total=", str(total_loop))
                    z = mu + torch.randn_like(mu) * perturb_std

                    gen_seq = generate_peptide(
                        decoder,
                        z,
                        tgt_func,
                        tgt_len,
                        temperature=temperature,
                        mode=mode,
                        top_k=top_k
                    )

                    rows.append({
                        "prototype_sequence": seq,
                        "generated_sequence": gen_seq,
                        "original_func": func,
                        "target_func": tgt_func,
                        "prototype_length": proto_len,
                        "target_length": tgt_len,
                        "length_delta": tgt_len - proto_len
                    })

                    fasta_entries.append(
                        f">proto{pid}_origF{func}_tgtF{tgt_func}_L{tgt_len}\n{gen_seq}"
                    )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    with open(fasta_path, "w") as f:
        f.write("\n".join(fasta_entries))

    print(
        f"✅ Length-window analogues saved to:\n"
        f"  CSV   → {csv_path}\n"
        f"  FASTA → {fasta_path}"
    )


# -----------------------------
# Main workflow
# -----------------------------
def main():
    dir_path = "/work/idoerg/priyanka/plum/generator/"
    TRAIN_CSV = dir_path + "train_test/train.csv"
    TEST_CSV = dir_path + "train_test/test.csv"

    MODEL_DIR = dir_path + "models"
    RANDOM_FASTA = dir_path + "generated_peptides/random_generation_baseline_2_lstm.fasta"
    OUTPUT_FASTA = dir_path + "generated_peptides/analogues_baseline_2_lstm.fasta"
    OUTPUT_CSV = dir_path + "generated_peptides/analogues_generation_baseline_2_lstm.csv"

    os.makedirs(os.path.dirname(OUTPUT_FASTA), exist_ok=True)

    # -----------------------------
    # Train LSTM CVAE on train.csv
    # -----------------------------
    print("Training CVAE LSTM model...")
    encoder, decoder, train_dataset = train_cvae_lstm(
        TRAIN_CSV, epochs=500, latent_dim=16, model_dir=MODEL_DIR
    )
    print("✅ Training completed.")
    # -----------------------------
    # Generate random peptides
    # -----------------------------
    print("Generating random peptides...")
    generate_random_peptides_lstm(
        decoder, latent_dim=16, seq_len=train_dataset.max_len,
        output_fasta=RANDOM_FASTA, temperature=0.8, peptides_per_func=330
    )

    # -----------------------------
    # Load test dataset for analogue generation
    # -----------------------------
    print("Loading test dataset...")
    df_test = pd.read_csv(TEST_CSV)
    peptides_test = df_test['sequence'].tolist()
    y_f_test = df_test['mic_class_binary'].astype(int).tolist()
    y_s_test = [len(p) for p in peptides_test]

    # -----------------------------
        # Length-window analogue generation
        # -----------------------------
        #print("Generating length-window analogue peptides...")

    csv_out_lenwin = (
        dir_path + "generated_peptides/analogues_lenwin_baseline_2_lstm_AR_200_2.csv"
    )
    fasta_out_lenwin = (
        dir_path + "generated_peptides/analogues_lenwin_baseline_2_lstm_AR.fasta"
    )

    generate_length_window_analogues(
        peptides=peptides_test,
        y_f=y_f_test,
        encoder=encoder,
        decoder=decoder,
        latent_dim=16,
        csv_path=csv_out_lenwin,
        fasta_path=fasta_out_lenwin,
        n=10,                 # analogues per (func, length)
        window=7,            
        min_len=5,           # hard lower bound
        max_len=35,          # hard upper bound
        perturb_std=0.01,
        temperature=1.0,
        mode="multinomial",
        top_k=None
    )

   
if __name__ == "__main__":
    main()
