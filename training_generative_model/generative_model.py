# csvae_peptide_ver_7_trigram_zprior.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# -----------------------------
# Constants
# -----------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYZ"  # includes stop token 'Z'
PAD_TOKEN = "-"
START_TOKEN = "#"

NUM_AMINO_ACIDS = len(AMINO_ACIDS) + 2  # + PAD + START
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX[PAD_TOKEN] = len(AMINO_ACIDS)
AA_TO_IDX[START_TOKEN] = len(AMINO_ACIDS) + 1
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

# -----------------------------
# Length bins
# -----------------------------
LENGTH_BINS = [(5,10), (11,15), (16,20), (21,25), (26,35)]
NUM_LENGTH_BINS = len(LENGTH_BINS)
BIN_MEANS = [np.mean(low_high) for low_high in LENGTH_BINS]

def length_to_bin(length):
    for i, (low, high) in enumerate(LENGTH_BINS):
        if low <= length <= high:
            vec = np.zeros(NUM_LENGTH_BINS, dtype=np.float32)
            vec[i] = 1.0
            return vec
    vec = np.zeros(NUM_LENGTH_BINS, dtype=np.float32)
    vec[-1] = 1.0
    return vec

# -----------------------------
# Dataset
# -----------------------------
class PeptideDataset_LSTM(Dataset):
    def __init__(self, sequences, functions, max_len=35):
        self.sequences = sequences
        self.functions = functions
        self.lengths = [len(seq.replace('-', '').replace('Z','').replace(START_TOKEN,'')) for seq in sequences]
        self.max_len = max_len

    def one_hot_encode(self, seq):
        mat = np.zeros((self.max_len, NUM_AMINO_ACIDS), dtype=np.float32)
        for i, aa in enumerate(seq[:self.max_len]):
            idx = AA_TO_IDX.get(aa, AA_TO_IDX[PAD_TOKEN])
            mat[i, idx] = 1.0
        return mat

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = self.one_hot_encode(seq)
        func = np.array([self.functions[idx]], dtype=np.float32)
        y_len = length_to_bin(self.lengths[idx])
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(func, dtype=torch.float32), \
               torch.tensor(y_len, dtype=torch.float32)

# -----------------------------
# CSVAE LSTM Model with z-only decoder and trigram-based prior
# -----------------------------
class PeptideCSVAE_LSTM(nn.Module):
    def __init__(self, seq_len=35, z_dim=32, w_dim=16, v_dim=8,
                 hidden_dim=128, cond_dim=1, length_dim=NUM_LENGTH_BINS,
                 lstm_layers=1, embed_dim=16, teacher_forcing_ratio=0.5):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = NUM_AMINO_ACIDS
        self.z_dim, self.w_dim, self.v_dim = z_dim, w_dim, v_dim
        self.cond_dim, self.length_dim = cond_dim, length_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.start_idx = AA_TO_IDX[START_TOKEN]

        # -----------------
        # Encoders
        # -----------------
        self.encoder_x_to_z = nn.LSTM(self.input_dim, hidden_dim, lstm_layers,
                                      batch_first=True, bidirectional=True)
        self.mu_x_to_z = nn.Linear(hidden_dim*2, z_dim)
        self.logvar_x_to_z = nn.Linear(hidden_dim*2, z_dim)

        self.encoder_xy_to_w = nn.LSTM(self.input_dim + cond_dim, hidden_dim,
                                       lstm_layers, batch_first=True, bidirectional=True)
        self.mu_xy_to_w = nn.Linear(hidden_dim*2, w_dim)
        self.logvar_xy_to_w = nn.Linear(hidden_dim*2, w_dim)

        self.encoder_y_to_w = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_y_to_w = nn.Linear(hidden_dim, w_dim)
        self.logvar_y_to_w = nn.Linear(hidden_dim, w_dim)

        self.encoder_xylen_to_v = nn.LSTM(self.input_dim + length_dim, hidden_dim,
                                          lstm_layers, batch_first=True, bidirectional=True)
        self.mu_xylen_to_v = nn.Linear(hidden_dim*2, v_dim)
        self.logvar_xylen_to_v = nn.Linear(hidden_dim*2, v_dim)

        self.encoder_ylen_to_v = nn.Sequential(
            nn.Linear(length_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_ylen_to_v = nn.Linear(hidden_dim, v_dim)
        self.logvar_ylen_to_v = nn.Linear(hidden_dim, v_dim)

        # -----------------
        # Auxiliary predictors
        # -----------------
        self.v_to_len = nn.Sequential(
            nn.Linear(v_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, length_dim)
        )
        self.w_to_func = nn.Sequential(
            nn.Linear(w_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cond_dim),
            nn.Sigmoid()
        )

        # -----------------
        # Decoder (full)
        # -----------------
        self.decoder_lstm = nn.LSTM(self.input_dim + z_dim + w_dim + v_dim,
                                    hidden_dim, lstm_layers, batch_first=True)
        self.out_x = nn.Linear(hidden_dim, self.input_dim)

        # -----------------
        # Separate decoder for z-only reconstruction
        # -----------------
        self.decoder_z_lstm = nn.LSTM(self.input_dim + z_dim,
                                      hidden_dim, lstm_layers, batch_first=True)
        self.out_x_z = nn.Linear(hidden_dim, self.input_dim)

        # -----------------
        # Trigram prior for z
        # -----------------
        self.trigram_embedding = nn.Embedding(NUM_AMINO_ACIDS, embed_dim)
        self.trigram_to_mu = nn.Linear(embed_dim, z_dim)
        self.trigram_to_logvar = nn.Linear(embed_dim, z_dim)

        # adversarial predictor from z (to encourage z to be uninformative about func)
        self.decoder_z_to_func = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cond_dim),
            nn.Sigmoid()
        )
        # adversarial predictor from z (to encourage z to be uninformative about func)
        self.decoder_z_to_len = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cond_dim)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim, device=device)
        return h, c

    def one_hot_encode(self, seq):
        mat = np.zeros((self.seq_len, self.input_dim), dtype=np.float32)
        for i, aa in enumerate(seq[:self.seq_len]):
            idx = AA_TO_IDX.get(aa, AA_TO_IDX[PAD_TOKEN])
            mat[i, idx] = 1.0
        return mat

    # -----------------
    # Encoder functions
    # -----------------
    def q_z(self, x):
        _, (h_n, _) = self.encoder_x_to_z(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.mu_x_to_z(h), self.logvar_x_to_z(h)

    def q_w(self, x, y):
        y_b = y.unsqueeze(1).repeat(1, self.seq_len, 1)
        xy = torch.cat([x, y_b], dim=2)
        _, (h_n, _) = self.encoder_xy_to_w(xy)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.mu_xy_to_w(h), self.logvar_xy_to_w(h)

    def q_v(self, x, y_len):
        ylen_b = y_len.unsqueeze(1).repeat(1, self.seq_len, 1)
        xylen = torch.cat([x, ylen_b], dim=2)
        _, (h_n, _) = self.encoder_xylen_to_v(xylen)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.mu_xylen_to_v(h), self.logvar_xylen_to_v(h)

    def p_w_prior(self, y):
        h = self.encoder_y_to_w(y)
        return self.mu_y_to_w(h), self.logvar_y_to_w(h)

    def p_v_prior(self, y_len):
        h = self.encoder_ylen_to_v(y_len)
        return self.mu_ylen_to_v(h), self.logvar_ylen_to_v(h)

    # -----------------
    # Decoder functions
    # -----------------
    def decode_full(self, z, w, v, seq=None, teacher_forcing=False, max_len=None):
        batch_size = z.size(0)
        device = z.device
        input_dim = self.input_dim
        h, c = self.init_hidden(batch_size, device)

        start_token = torch.zeros(batch_size, 1, input_dim, device=device)
        start_token[:, 0, self.start_idx] = 1.0
        input_t = start_token

        x_logits = []
        seq_len = max_len if max_len is not None else self.seq_len

        for t in range(seq_len):
            lstm_input = torch.cat([input_t, z.unsqueeze(1), w.unsqueeze(1), v.unsqueeze(1)], dim=2)
            
            output, (h, c) = self.decoder_lstm(lstm_input, (h, c))
            logits_t = self.out_x(output)
            x_logits.append(logits_t)

            if teacher_forcing and seq is not None:
                input_t = seq[:, t].unsqueeze(1)
            else:
                token_idx = logits_t.argmax(dim=2)
                input_t = F.one_hot(token_idx.squeeze(1), num_classes=input_dim).float().unsqueeze(1)

        return torch.cat(x_logits, dim=1)

    def decode_z_only(self, z, seq=None, teacher_forcing=False, max_len=None):
        batch_size = z.size(0)
        device = z.device
        input_dim = self.input_dim
        h, c = self.init_hidden(batch_size, device)

        start_token = torch.zeros(batch_size, 1, input_dim, device=device)
        start_token[:, 0, self.start_idx] = 1.0
        input_t = start_token

        x_logits = []
        seq_len = max_len if max_len is not None else self.seq_len

        for t in range(seq_len):
            lstm_input = torch.cat([input_t, z.unsqueeze(1)], dim=2)
            output, (h, c) = self.decoder_z_lstm(lstm_input, (h, c))
            logits_t = self.out_x_z(output)
            x_logits.append(logits_t)

            if teacher_forcing and seq is not None:
                input_t = seq[:, t].unsqueeze(1)
            else:
                token_idx = logits_t.argmax(dim=2)
                input_t = F.one_hot(token_idx.squeeze(1), num_classes=input_dim).float().unsqueeze(1)

        return torch.cat(x_logits, dim=1)

    # -----------------
    # Forward
    # -----------------
    def forward(self, x, y, y_len, teacher_forcing=False):
        mu_z, logvar_z = self.q_z(x)
        mu_w_enc, logvar_w_enc = self.q_w(x, y)
        mu_v_enc, logvar_v_enc = self.q_v(x, y_len)

        mu_w_prior, logvar_w_prior = self.p_w_prior(y)
        mu_v_prior, logvar_v_prior = self.p_v_prior(y_len)

        z = self.reparameterize(mu_z, logvar_z)
        w = self.reparameterize(mu_w_enc, logvar_w_enc)
        v = self.reparameterize(mu_v_enc, logvar_v_enc)

        x_logits = self.decode_full(z, w, v, seq=x, teacher_forcing=teacher_forcing)
        x_logits_zonly = self.decode_z_only(z, seq=x, teacher_forcing=teacher_forcing)

        y_pred_len = self.v_to_len(v)
        y_pred_func = self.w_to_func(w)

        return {
            "x_logits": x_logits,
            "x_logits_zonly": x_logits_zonly,
            "z": z, "w": w, "v": v,
            "mu_z": mu_z, "logvar_z": logvar_z,
            "mu_w_enc": mu_w_enc, "logvar_w_enc": logvar_w_enc,
            "mu_w_prior": mu_w_prior, "logvar_w_prior": logvar_w_prior,
            "mu_v_enc": mu_v_enc, "logvar_v_enc": logvar_v_enc,
            "mu_v_prior": mu_v_prior, "logvar_v_prior": logvar_v_prior,
            "y_pred_len": y_pred_len,
            "y_pred_func": y_pred_func
        }

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu_q, logvar_q, mu_p, logvar_p, dim=None):
        kl = 0.5 * (logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p).pow(2)) / torch.exp(logvar_p) - 1)
        return kl.sum(dim=dim).mean() if dim is not None else kl.sum(dim=1).mean()
