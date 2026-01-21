import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# -----------------------------
# Amino acids
# -----------------------------
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# -----------------------------
# CVAE Models (must match original definitions)
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
# Dataset helper
# -----------------------------
class PeptideDataset:
    def __init__(self, peptides, y_f):
        self.peptides = peptides
        self.y_f = y_f
        self.y_s = [len(p) for p in peptides]

def main():
    dir_path="/content/drive/MyDrive/Colab Notebooks/PLUM/"
    MODEL_DIR = dir_path + "models/baseline_1"
    RANDOM_FASTA = dir_path + "generated_peptides/random_generation_baseline_1.fasta"
    ANALOGUE_CSV = dir_path + "generated_peptides/analogues_generation_baseline_1.csv"
    TEST_CSV = dir_path + "train_test/test.csv"
    os.makedirs(os.path.dirname(RANDOM_FASTA), exist_ok=True)

    # -----------------------------
    # Load config
    # -----------------------------
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)
    seq_len = config["seq_len"]
    latent_dim = config["latent_dim"]
    hidden_dim = 128
    max_len = seq_len
    # -----------------------------
    # Load models
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder = torch.load(os.path.join(MODEL_DIR, "cvae_encoder_full.pth"), map_location=device)
    # decoder = torch.load(os.path.join(MODEL_DIR, "cvae_decoder_full.pth"), map_location=device)
    # encoder.to(device).eval()
    # decoder.to(device).eval()
    encoder = Encoder(
        input_dim=20,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        seq_len=seq_len,
        cond_dim=2
    ).to(device)

    decoder = Decoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        output_dim=20,
        seq_len=seq_len,
        cond_dim=2
    ).to(device)

    encoder = torch.load(
        os.path.join(MODEL_DIR, "cvae_encoder_full.pth"),
        map_location=device,
        weights_only=False
    )

    decoder = torch.load(
        os.path.join(MODEL_DIR, "cvae_decoder_full.pth"),
        map_location=device,
        weights_only=False
    )


    encoder.eval()
    decoder.eval()

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
        n=100,               # per (func, length)
        window=5,          # ±5 AA
        min_len=5,
        max_len=35,
        temperature=1.0,
        perturb_std=0.3,
        alpha=0.4,
        mode="multinomial",
        top_k=None
    )

if __name__ == "__main__":
    main()