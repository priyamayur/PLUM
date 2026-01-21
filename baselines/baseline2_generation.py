import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

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
# Encoder & Decoder definitions
# -----------------------------
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim=VOCAB_SIZE, hidden_dim=128, latent_dim=16, cond_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + cond_dim, latent_dim)

    def forward(self, x, cond, lengths):
        from torch.nn.utils.rnn import pack_padded_sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = h[-1]
        h = torch.cat([h, cond], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

class DecoderLSTM(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=128, cond_dim=2,
                 embedding_dim=64, max_len=35):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(VOCAB_SIZE, embedding_dim, padding_idx=PAD_IDX)
        self.latent_to_hidden = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, VOCAB_SIZE)

    def forward(self, z, cond, targets=None, teacher_forcing_ratio=0.0):
        B = z.size(0)
        device = z.device

        h = self.latent_to_hidden(torch.cat([z, cond], dim=1)).unsqueeze(0)
        c = torch.zeros_like(h)

        inputs = torch.full((B,), SOS_IDX, dtype=torch.long, device=device)
        emb = self.embedding(inputs).unsqueeze(1)

        outputs = []

        for t in range(self.max_len):
            out, (h, c) = self.lstm(emb, (h, c))
            logits = self.fc_out(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_token = targets[:, t]
            else:
                next_token = logits.argmax(dim=1)

            emb = self.embedding(next_token).unsqueeze(1)

        return torch.cat(outputs, dim=1)

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
# Main
# -----------------------------
def main():
    dir_path="/content/drive/MyDrive/Colab Notebooks/PLUM/"
    model_dir = dir_path + "models/baseline_2_lstm_AR"
    test_csv = dir_path + "train_test/test.csv"

    # -----------------------------
    # Load metadata
    # -----------------------------
    with open(os.path.join(model_dir, "dataset_meta.json"), "r") as f:
        meta = json.load(f)
    max_len = meta["max_len"]
    latent_dim = meta["latent_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load models
    # -----------------------------
    # encoder = torch.load(os.path.join(model_dir, "encoder_full.pth"), map_location=device)
    # decoder = torch.load(os.path.join(model_dir, "decoder_full.pth"), map_location=device)
    # encoder.to(device).eval()
    # decoder.to(device).eval()

    encoder = torch.load(
    os.path.join(model_dir, "encoder_full.pth"),
    map_location=device,
    weights_only=False
    )

    decoder = torch.load(
        os.path.join(model_dir, "decoder_full.pth"),
        map_location=device,
        weights_only=False
    )

    encoder.eval().to(device)
    decoder.eval().to(device)

    df_test = pd.read_csv(test_csv)
    peptides = df_test.sequence.tolist()
    y_f = df_test.mic_class_binary.tolist()
    y_s = [len(s) for s in peptides]

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
        peptides=peptides,
        y_f=y_f,
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        csv_path=csv_out_lenwin,
        fasta_path=fasta_out_lenwin,
        n=10,                 # analogues per (func, length)
        window=5,            # ±5 AA from prototype length
        min_len=5,           # hard lower bound
        max_len=35,          # hard upper bound
        perturb_std=0.01,
        temperature=1.0,
        mode="multinomial",
        top_k=None
    )

if __name__ == "__main__":
    main()