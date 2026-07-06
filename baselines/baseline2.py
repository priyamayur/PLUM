# cvae_lstm_autoregressive.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import pandas as pd

# -----------------------------
# Amino acids + special tokens
# -----------------------------
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
PAD_TOKEN = '<PAD>'
EOS_TOKEN = '<EOS>'
SOS_TOKEN = '<SOS>'

ALL_TOKENS = list(AMINO_ACIDS) + [PAD_TOKEN, EOS_TOKEN, SOS_TOKEN]

AA_TO_IDX = {aa: i for i, aa in enumerate(ALL_TOKENS)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

PAD_IDX = AA_TO_IDX[PAD_TOKEN]
EOS_IDX = AA_TO_IDX[EOS_TOKEN]
SOS_IDX = AA_TO_IDX[SOS_TOKEN]

VOCAB_SIZE = len(ALL_TOKENS)

# -----------------------------
# Dataset
# -----------------------------
class PeptideDataset(Dataset):
    def __init__(self, peptides, y_f, y_s=None):
        self.peptides = peptides
        self.y_f = np.array(y_f, dtype=np.int64)
        # Ensure max_len has room for the EOS token
        self.max_len = max(len(p) for p in peptides) + 1
        
    def __len__(self):
            return len(self.peptides)
        
    def __getitem__(self, idx):
        peptide = self.peptides[idx]
        length = len(peptide)
        x = np.zeros((self.max_len, VOCAB_SIZE), dtype=np.float32)
        target = np.full((self.max_len,), PAD_IDX, dtype=np.int64)

        for i, aa in enumerate(peptide):
            x[i, AA_TO_IDX[aa]] = 1.0
            target[i] = AA_TO_IDX[aa]

        # Always add EOS
        target[length] = EOS_IDX
        
        # We tell the encoder to look at the AA sequence + the EOS token
        effective_len = length + 1
        cond = np.array([self.y_f[idx], length / (self.max_len - 1)], dtype=np.float32)
        return x, target, cond, effective_len

# -----------------------------
# Encoder (packed LSTM)
# -----------------------------
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim=VOCAB_SIZE, hidden_dim=128, latent_dim=16, cond_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + cond_dim, latent_dim)

    def forward(self, x, cond, lengths):
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        h = h[-1]
        h = torch.cat([h, cond], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

# -----------------------------
# Decoder (autoregressive)
# -----------------------------
class DecoderLSTM(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=128, cond_dim=2, embedding_dim=64, max_len=35):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_IDX)
        
        # New: The LSTM now takes (Embedding + Latent + Condition)
        self.lstm_input_dim = embedding_dim + latent_dim + cond_dim
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, batch_first=True)
        
        self.latent_to_hidden = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, z, cond, targets=None, teacher_forcing_ratio=1.0):
        B = z.size(0)
        device = z.device
        
        # Initial state still comes from z and cond
        z_cond = torch.cat([z, cond], dim=1) # Shape: [B, latent_dim + cond_dim]
        h = self.latent_to_hidden(z_cond).unsqueeze(0)
        c = torch.zeros_like(h)

        inputs = torch.full((B,), SOS_IDX, dtype=torch.long, device=device)
        outputs = []

        for t in range(self.max_len):
            # 1. Get embedding for current token
            emb = self.embedding(inputs).unsqueeze(1) # [B, 1, embedding_dim]
            
            # 2. Concatenate z_cond to the embedding at every step
            # We expand z_cond to match the sequence length (which is 1 here)
            z_cond_expanded = z_cond.unsqueeze(1) # [B, 1, latent_dim + cond_dim]
            lstm_in = torch.cat([emb, z_cond_expanded], dim=-1) # [B, 1, lstm_input_dim]

            # 3. Pass through LSTM
            out, (h, c) = self.lstm(lstm_in, (h, c))
            
            logits = self.fc_out(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            # Decide next token (Teacher Forcing vs. Sampling)
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                inputs = targets[:, t]
            else:
                inputs = logits.argmax(dim=1)

        return torch.cat(outputs, dim=1)

# -----------------------------
# VAE utils
# -----------------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))

# -----------------------------
# Training
# -----------------------------
def train_cvae_lstm(csv_path, epochs=300, batch_size=16, lr=1e-3, latent_dim=16, model_dir="models"):
    df = pd.read_csv(csv_path)
    dataset = PeptideDataset(df.sequence.tolist(), df.mic_class_binary.tolist())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderLSTM(latent_dim=latent_dim).to(device)
    decoder = DecoderLSTM(latent_dim=latent_dim, max_len=dataset.max_len).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(epochs):
        encoder.train(); decoder.train()
        total_loss = 0
        tf_ratio = max(0.1, 1.0 - epoch / 200)

        for x, target, cond, lengths in loader:
            x, target, cond = x.to(device), target.to(device), cond.to(device)
            optimizer.zero_grad()

            mu, logvar = encoder(x, cond, lengths)
            z = reparameterize(mu, logvar)
            logits = decoder(z, cond, target, tf_ratio)

            rec = ce_loss(logits.view(-1, VOCAB_SIZE), target.view(-1))
            kl = kl_divergence(mu, logvar)
            loss = rec + 0.01 * kl

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss {total_loss/len(loader):.4f} | rec {rec.item():.4f} | kl {kl.item():.4f}")

    os.makedirs(model_dir, exist_ok=True)
    torch.save(encoder.state_dict(), f"{model_dir}/encoder.pth")
    torch.save(decoder.state_dict(), f"{model_dir}/decoder.pth")

    return encoder, decoder, dataset

# -----------------------------
# Generation
# -----------------------------
def generate_peptide(decoder, z, func, target_length, temperature=1.0):
    device = z.device
    decoder.eval()
    with torch.no_grad():
        cond = torch.tensor([[func, target_length / (decoder.max_len - 1)]], 
                            dtype=torch.float32, device=device)
        
        # Instead of calling decoder forward, we should ideally 
        # let the loop inside the decoder break on EOS_IDX.
        # For now, your current decoder returns the full max_len. 
        # Let's extract tokens until EOS:
        logits = decoder(z, cond, targets=None, teacher_forcing_ratio=0.0)
        probs = F.softmax(logits / temperature, dim=-1)

        seq = []
        for t in range(decoder.max_len):
            idx = torch.multinomial(probs[0, t], 1).item()
            if idx == EOS_IDX or len(seq) >= target_length:
                break
            if idx < len(AMINO_ACIDS):
                seq.append(AMINO_ACIDS[idx])
    return "".join(seq)

# -----------------------------
# Random generation
# -----------------------------
def generate_random_peptides(decoder, latent_dim, max_len, fasta_path,
                             n=10, temperature=1.0):
    device = next(decoder.parameters()).device
    os.makedirs(os.path.dirname(fasta_path), exist_ok=True)

    with open(fasta_path, "w") as f:
        idx = 1
        for func in [0, 1]:
            for length in range(5, max_len + 1):
                for _ in range(n):
                    z = torch.randn(1, latent_dim, device=device)
                    pep = generate_peptide(decoder, z, func, length, temperature)
                    f.write(f">pep_{idx}_func{func}_len{length}\n{pep}\n")
                    idx += 1

# -----------------------------
# Main
# -----------------------------
def main():
    base = "/work/idoerg/priyanka/plum/generator/"
    TRAIN = base + "train_test/train.csv"
    TEST = base + "train_test/test.csv"

    encoder, decoder, dataset = train_cvae_lstm(TRAIN, epochs=100)

    print("Training completed.")
    model_dir = base + "models/baseline_2_lstm_AR_corrected_may_2026"
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the full models (architecture + weights)
    torch.save(encoder, f"{model_dir}/encoder_full.pth")
    torch.save(decoder, f"{model_dir}/decoder_full.pth")

    # Optional: also save state_dicts as backup
    torch.save(encoder.state_dict(), f"{model_dir}/encoder_state.pth")
    torch.save(decoder.state_dict(), f"{model_dir}/decoder_state.pth")

    # Save dataset info (max_len is needed for generation)
    import json
    metadata = {"max_len": dataset.max_len, "latent_dim": 16}  # update latent_dim if different
    with open(f"{model_dir}/dataset_meta.json", "w") as f:
        json.dump(metadata, f)

    print(f"Models and metadata saved in '{model_dir}'")


    print("Starting random peptide generation...")
    generate_random_peptides(
        decoder, 16, dataset.max_len,
        base + "generated_peptides/random_generation_baseline_3_lstm_AR.fasta", n=330
    )
    

   

if __name__ == "__main__":
    main()
