import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
import torch
import joblib
from collections import defaultdict
from Bio import SeqIO
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load AMP classifier ---
AMP_CLASSIFIER_PATH = "./saved_models_AMP_classifier/best_model/classifier.pkl"
amp_classifier = joblib.load(AMP_CLASSIFIER_PATH)
# --- Load MIC classifier ---
MIC_CLASSIFIER_PATH = "./saved_models_MIC/best_model/classifier.pkl"
mic_classifier = joblib.load(MIC_CLASSIFIER_PATH)

# --- Load ProtT5 Encoder ---
ENCODER_INFO_PATH = "./saved_models_AMP_classifier/best_model/encoder_info.json"
with open(ENCODER_INFO_PATH) as f:
    encoder_info = json.load(f)
    
encoder_name = encoder_info["encoder_name"]
encoder = T5EncoderModel.from_pretrained(encoder_name).to(DEVICE).eval()
tokenizer = T5Tokenizer.from_pretrained(encoder_name, do_lower_case=False)

@torch.inference_mode()
def generate_embeddings(sequences, model, tokenizer, batch_size=64):
    sequences = [" ".join(list(seq.upper())) for seq in sequences]
    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        outputs = model(**tokens)
        hidden_states = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)

# --- Load prototypes ---
prototypes_df = pd.read_csv("./analogue/test.csv")  # sequence,mic_class_binary,record_id

# --- Methods and corresponding generated CSVs ---
methods = {
    "HydrAMP": "./analogue/HydrAMP_analogues_test.csv",
    "Joker": "./analogue/joker_analogues_test.csv"
}
methods = {
    "HydrAMP": "./analogue/HydrAMP_analogues_test_200.csv",
    "Joker": "./analogue/joker_analogues_test.csv",
    "Baseline1": "./analogue/Baseline1_analogues_test_target_1.csv",
    "Baseline2": "./analogue/Baseline2_analogues_test_target_1.csv",
    "PLUM": "./analogue/PLUM_analogues_test_target_1.csv" 
}


from Bio import pairwise2

# ------------------------
# Alignment functions
# ------------------------

def global_identity(seq1, seq2):
    """Needleman-Wunsch global alignment identity fraction"""
    align = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    matches = sum(a==b for a,b in zip(align.seqA, align.seqB))
    return matches / max(len(seq1), len(seq2))

def local_identity(seq1, seq2):
    """Smith-Waterman local alignment identity fraction"""
    align = pairwise2.align.localxx(seq1, seq2, one_alignment_only=True)[0]
    matches = sum(a==b for a,b in zip(align.seqA, align.seqB))
    return matches / min(len(seq1), len(seq2))

AMP_THRESHOLD = 0.5

MIC_THRESHOLD = 0.5

for method_name, gen_csv in methods.items():
    print(f"\n=== Method: {method_name} ===")

    generated_df = pd.read_csv(gen_csv)
    generated_df.columns = generated_df.columns.str.strip()

    proto_to_analogues = {
        proto: [seq for seq in set(analogues) if seq != proto]
        for proto, analogues in generated_df.groupby("parent_sequence")["generated_sequence"]
    }

    class1_prototypes = prototypes_df[prototypes_df["mic_class_binary"] == 1]["sequence"].tolist()
    class0_prototypes = prototypes_df[prototypes_df["mic_class_binary"] == 0]["sequence"].tolist()

    for mic_class, proto_list in [("Class 1 (mic=1)", class1_prototypes),
                                  ("Class 0 (mic=0)", class0_prototypes)]:

        pooled_amp_seqs = []
        pooled_potent_seqs = []

        proto_amp_fractions = []
        proto_potent_fractions = []
        proto_global_sim = []
        proto_local_sim = []
        valid_proto_count = 0

        for p in proto_list:
            if p not in proto_to_analogues:
                continue

            seqs = proto_to_analogues[p]
            if len(seqs) == 0:
                continue

            valid_proto_count += 1

            # --- Embed all analogues for this prototype ---
            Xp = generate_embeddings(seqs, encoder, tokenizer)

            # --- AMP prediction ---
            amp_probs = amp_classifier.predict_proba(Xp)[:, 1]
            amp_mask = amp_probs >= AMP_THRESHOLD
            amp_seqs = [s for s, m in zip(seqs, amp_mask) if m]

            proto_amp_fractions.append(np.mean(amp_mask))

            if len(amp_seqs) == 0:
                proto_potent_fractions.append(0.0)
                proto_global_sim.append(0.0)
                proto_local_sim.append(0.0)
                continue

            pooled_amp_seqs.extend(amp_seqs)

            # --- MIC prediction ONLY on AMP-positive peptides ---
            X_amp = Xp[amp_mask]
            mic_probs = mic_classifier.predict_proba(X_amp)[:, 1]
            potent_mask = mic_probs >= MIC_THRESHOLD

            proto_potent_fractions.append(np.mean(potent_mask))
            pooled_potent_seqs.extend(
                [s for s, m in zip(amp_seqs, potent_mask) if m]
            )

            # --- Compute alignment-based similarities ---
            global_sims = [global_identity(p, s) for s in amp_seqs]
            local_sims  = [local_identity(p, s) for s in amp_seqs]

            proto_global_sim.append(np.mean(global_sims))
            proto_local_sim.append(np.mean(local_sims))

        if not pooled_amp_seqs:
            print(f"{mic_class}: No AMP-positive peptides found")
            continue

        # --- Convert to arrays ---
        proto_amp_fractions = np.array(proto_amp_fractions)
        proto_potent_fractions = np.array(proto_potent_fractions)
        proto_global_sim = np.array(proto_global_sim)
        proto_local_sim = np.array(proto_local_sim)

        print(f"\n{mic_class}:")
        print(f"  Total prototypes (class)         : {len(proto_list)}")
        print(f"  Prototypes with analogues        : {valid_proto_count}")
        print(f"  AMP-positive peptides (pooled)   : {len(pooled_amp_seqs)}")
        print(f"  Potent AMPs (pooled)             : {len(pooled_potent_seqs)}")
        print(f"  Pooled potent|AMP fraction       : {len(pooled_potent_seqs) / len(pooled_amp_seqs):.3f}")
        print(f"  Mean per-proto AMP fraction      : {proto_amp_fractions.mean():.3f}")
        print(f"  Mean per-proto potent|AMP fraction: {proto_potent_fractions.mean():.3f}")
        print(f"  Std per-proto potent|AMP fraction : {proto_potent_fractions.std():.3f}")
        print(f"  Mean per-proto global identity    : {proto_global_sim.mean():.3f}")
        print(f"  Std per-proto global identity     : {proto_global_sim.std():.3f}")
        print(f"  Mean per-proto local identity     : {proto_local_sim.mean():.3f}")
        print(f"  Std per-proto local identity      : {proto_local_sim.std():.3f}")
