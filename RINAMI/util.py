import copy
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from transformers import get_cosine_schedule_with_warmup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Bio.PDB import PPBuilder

def get_sequence_from_single_chain_pdb(pdb_path):
    """

    Args:
        sequences            : AA-seq list
        padding_value (float): default = 0.0

    Returns:
        np.ndarray: one-hot tensor (shape: (N, max_len, 20))
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ppb = PPBuilder()
    model = structure[0]
    chain = list(model.get_chains())[0]
    sequence = ppb.build_peptides(chain)[0].get_sequence()
    
    return str(sequence)


def aa_sequences_to_padded_onehot(sequences, aa_order="ACDEFGHIKLMNPQRSTVWY", padding_value=0.0):
    """
    Args:
        sequences            : AA-seq list
        padding_value (float): default = 0.0

    Returns:
        np.ndarray: one-hot tensor (shape: (N, max_len, 20))
    """
    aa_to_index = {aa: i for i, aa in enumerate(aa_order)}
    max_len = max(len(seq) for seq in sequences)
    num_aa = len(aa_order)
    
    batch_size = len(sequences)
    onehot_tensor = np.full((batch_size, max_len, num_aa), padding_value, dtype=np.float32)

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if aa in aa_to_index:
                onehot_tensor[i, j, aa_to_index[aa]] = 1.0
            # Unknown residues keep the padding value.
    return torch.from_numpy(np.asarray(onehot_tensor, dtype=np.float32))


#######################################################################
# Organize the list of Structural representations into a padded batch #
#######################################################################
def pad_feature_matrices(feature_list, padding_value=0.0):
    """
    Args:
        Structural representations : List of features (np.ndarray : shape = (seq_len, feature_dim))
        padding_value (float)      : default = 0.0

    Returns:
        torch.tensor: shape = (batch_size, max_seq_len, feature_dim)) 
    """
    batch_size = len(feature_list)
    max_len = max(mat.shape[0] for mat in feature_list)
    feat_dim = feature_list[0].shape[1]

    padded_tensor = np.full((batch_size, max_len, feat_dim), padding_value, dtype=np.float32)

    for i, mat in enumerate(feature_list):
        seq_len = mat.shape[0]
        padded_tensor[i, :seq_len, :] = mat

    return torch.from_numpy(padded_tensor)





##################################################################
# Train-time sub-sampling ratio                                  #
# 1.0 uses the full training dataset.                            #
# 0.8 randomly samples 80% of the training dataset at each epoch.#
# Validation, test, and export always use all samples.           #
##################################################################
TRAIN_SAMPLE_FRACTION = 1.0
ESM_SIZE = 320


#########################################################################
# Setting of the model ensemble comprising models trained on each split #
#########################################################################
DEFAULT_ENSEMBLE_ROOT = "../pth/lowest_performing_models"
DEFAULT_ENSEMBLE_ROOT_TRAINED_BY_USER = "../pth/pth_RINAMI_trained"
DEFAULT_ENSEMBLE_SPLITS = (1, 2, 3)

#####################################################
# Setting of the amino acid order for plot making   #
#####################################################
AA_ORDER_CANONICAL = "ACDEFGHIKLMNPQRSTVWY"
AA_ORDER_PLOT = "QENHDRKTSAGMCLVIWYFP"
AA_REORDER_INDEX = [AA_ORDER_CANONICAL.index(aa) for aa in AA_ORDER_PLOT]

#############################################################
# Approximate maximum ASA values based on Tien et al. 2013. #
#############################################################
MAX_ASA = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "Q": 214.0, "E": 214.0, "G": 97.0,  "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}

#################################################
# Convering AA three letter into AA one letter  #
#################################################
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

########################################################
# Pick wild-type protein structures and calculate RSA  #
########################################################
PDB_CANDIDATE_ROOTS = [
    "../processed_data/Mega_predicted_structure_pdb",
    ]

def resolve_wt_pdb_path(clean_name):
    """
    Args:
        clean_name of the Mega-scale protein: Protein name without ".pdb", "|", and ":" 

    Returns:
        the exact wild-type pdb path
        
    ※This function is used in the process of "load_mega_test_and_val_wt_only."
    """
    candidates = []

    for root in PDB_CANDIDATE_ROOTS:
        exact = os.path.join(root, f"{clean_name}.pdb")
        if os.path.exists(exact):
            return exact

        hits = glob.glob(os.path.join(root, f"{clean_name}*.pdb"))
        if len(hits) > 0:
            return hits[0]
        candidates.append(exact)

    raise FileNotFoundError(
        f"No pdb file found for {clean_name}. "
        f"Tried roots: {PDB_CANDIDATE_ROOTS}"
    )

def extract_residue_info_and_rsa(pdb_path):
    """
    Args:
        pdb_path

    Returns:
        residue_infos: {"aa": aa_seq, "resseq": list of residue numbers, "rsa": list of RSA-values, "buried": binary list of burial infomation} 
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)

    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    residue_infos = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue

                resname = residue.get_resname().upper()
                aa = THREE_TO_ONE.get(resname, "X")
                if aa not in MAX_ASA:
                    continue

                sasa = float(getattr(residue, "sasa", 0.0))
                rsa = sasa / MAX_ASA[aa]
                resseq = int(residue.id[1])

                residue_infos.append({
                    "aa": aa,
                    "resseq": resseq,
                    "rsa": rsa,
                    "buried": rsa < 0.16,
                })
        break

    if len(residue_infos) == 0:
        raise ValueError(f"No standard amino-acid residues found in {pdb_path}")

    return residue_infos

######################################################################
# Prepare the list of pairs of residue-number and AA for plot making #
######################################################################
def align_residue_info_to_sequence(seq, residue_infos, pdb_path):
    """
    Args:
        aa_sequence, residue_infos extracted by "extract_residue_info_and_rsa", pdb_path
    Process:
        Checking whether the lengthes of the aa_sequence and structure matches or not
    """
    
    pdb_seq = "".join(x["aa"] for x in residue_infos)

    if len(pdb_seq) == len(seq):
        return residue_infos

    print(
        f"[WARN] Sequence length mismatch for {pdb_path}: "
        f"seq_len={len(seq)}, pdb_len={len(pdb_seq)}. "
        f"Using min length and sequential fallback for overflow."
    )

    min_len = min(len(seq), len(residue_infos))
    aligned = residue_infos[:min_len]

    if len(seq) > min_len:
        next_resseq = aligned[-1]["resseq"] + 1 if len(aligned) > 0 else 1
        for i in range(min_len, len(seq)):
            aligned.append({
                "aa": seq[i],
                "resseq": next_resseq,
                "rsa": np.nan,
                "buried": False,
            })
            next_resseq += 1

    return aligned


def make_residue_labels(seq, residue_infos, valid_len):
    """
    Args:
        aa_sequence, residue_infos extracted by "extract_residue_info_and_rsa", valid_len(: length of aa_sequence initially input to RINAMI)

    Returns:
        Residue labels for the heatmap of residue-amino-acid-wise ΔG matrix: ["M1", "D2",...] 
    """
    labels = []
    buried_mask = []
    rsa_list = []
    residue_numbers = []

    for i in range(valid_len):
        aa_seq = seq[i]
        info = residue_infos[i] if i < len(residue_infos) else {
            "aa": aa_seq,
            "resseq": i + 1,
            "rsa": np.nan,
            "buried": False,
        }

        labels.append(f"{info['resseq']}{aa_seq}")
        buried_mask.append(bool(info["buried"]))
        rsa_list.append(float(info["rsa"]) if info["rsa"] == info["rsa"] else np.nan)
        residue_numbers.append(int(info["resseq"]))

    return labels, np.array(buried_mask, dtype=bool), np.array(rsa_list, dtype=float), np.array(residue_numbers, dtype=int)

#################################################
# Saving the residue-amino-acid-wise ΔG matrix  #
#################################################
def save_interpretability_heatmap(
    residue_amino_acid_wise_dG_mat,
    valid_len,
    seq,
    residue_infos,
    png_path,
    vmin=-1.,
    vmax=1.,
):
    valid_len = int(valid_len)
    residue_amino_acid_wise_dG_mat = residue_amino_acid_wise_dG_mat[:valid_len, :]
    residue_amino_acid_wise_dG_mat = residue_amino_acid_wise_dG_mat[:, AA_REORDER_INDEX].T  # (20, L)

    xticklabels, buried_mask, rsa_array, residue_numbers = make_residue_labels(
        seq=seq,
        residue_infos=residue_infos,
        valid_len=valid_len,
    )

    fig_w = max(10, valid_len * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    im = ax.imshow(
        residue_amino_acid_wise_dG_mat,
        aspect="auto",
        cmap="bwr",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(AA_ORDER_PLOT)))
    ax.set_yticklabels(list(AA_ORDER_PLOT), fontsize=15, fontweight='bold')

    ax.set_xticks(np.arange(valid_len))
    ax.set_xticklabels(xticklabels, rotation=90, fontsize=15, fontweight='bold')

    for tick, is_buried in zip(ax.get_xticklabels(), buried_mask):
        tick.set_color("black" if is_buried else "gray")


    ax.set_xlabel("Residue number + WT amino acid", fontsize=13, fontweight='bold')
    ax.set_ylabel("Amino acid", fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Partial ΔG Score [kcal/mol]", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return buried_mask, rsa_array, residue_numbers


##############################################################################################
# Removing ":" or "|" from the Mega-scale variant names,to evade errors on the command line  #
##############################################################################################
def rename_data(data_name_in_df):
    name = str(data_name_in_df)
    if ":" in name:
        name = name.replace(":", "_")
    if "|" in name:
        name = name.replace("|", "_")
    return name

##########################################
# Calculations of the evaluation metrics #
##########################################
def safe_corr(x, y): # Pearson's R
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def safe_spearman(x, y): # Spearman's R
    if len(x) < 2:
        return float("nan")
    res = spearmanr(x, y)
    val = getattr(res, "statistic", None)
    if val is None:
        val = getattr(res, "correlation", None)
    if val is None:
        try:
            val = res[0]
        except Exception:
            return float("nan")
    return float(val) if val == val else float("nan")


################################################
# Pos_weight for mitigating the data imbalance #
################################################
def compute_binary_pos_weight(labels):
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return torch.tensor(1.0, dtype=torch.float32, device=device), n_pos, n_neg

    pos_weight = n_neg / n_pos
    return torch.tensor(pos_weight, dtype=torch.float32, device=device), n_pos, n_neg


###################################################################################
# Random sampling of training data for each epoch, following the sample_fraction  #
###################################################################################
def get_sampled_indices(n_items, sample_fraction=1.0, shuffle=True):
    if n_items <= 0:
        return np.array([], dtype=np.int64)

    if sample_fraction is None:
        sample_fraction = 1.0

    sample_fraction = float(sample_fraction)
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")

    idx = np.arange(n_items)

    if sample_fraction < 1.0:
        n_pick = max(1, int(np.floor(n_items * sample_fraction)))
        idx = np.random.choice(idx, size=n_pick, replace=False)

    if shuffle:
        np.random.shuffle(idx)

    return idx.astype(np.int64)


def iterate_batches(dataset, batch_size, shuffle=True, sample_fraction=1.0):
    idx = get_sampled_indices(
        n_items=len(dataset["seqs"]),
        sample_fraction=sample_fraction,
        shuffle=shuffle,
    )

    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        batch = {
            "seqs": [dataset["seqs"][i] for i in batch_idx],
            "structs": [dataset["structs"][i] for i in batch_idx],
            "profiles": [dataset["profiles"][i] for i in batch_idx],
            "names": [dataset["names"][i] for i in batch_idx],
        }

        if "dgs" in dataset:
            batch["dgs"] = torch.tensor(
                [dataset["dgs"][i] for i in batch_idx],
                dtype=torch.float32,
                device=device,
            )
        if "fold" in dataset:
            batch["fold"] = torch.tensor(
                [dataset["fold"][i] for i in batch_idx],
                dtype=torch.float32,
                device=device,
            )

        yield batch


###################################################
# Loading data for the ΔG regression (Mega-scale) #
###################################################
def build_regression_dataset_from_csv(
    csv_path,
    struct_root,
    profile_root,
    name_col="name",
    seq_col="aa_seq",
    dg_col="dG_ML",
):
    df = pd.read_csv(csv_path, low_memory=False)

    seqs, dgs, structs, profiles, names = [], [], [], [], []
    for name, seq, dg in zip(df[name_col], df[seq_col], df[dg_col]):
        clean_name = rename_data(name).replace(".pdb", "").replace("__", "_")
        seqs.append(seq)
        dgs.append(float(dg))
        structs.append(f"{struct_root}/{clean_name}.pt")
        profiles.append(f"{profile_root}/{clean_name}_profile.npy")
        names.append(clean_name)

    return {
        "seqs": seqs,
        "dgs": dgs,
        "structs": structs,
        "profiles": profiles,
        "names": names,
    }


############################################################
# Loading data for the foldability prediction (Mega-scale) #
############################################################
def build_classification_dataset_from_csv(
    csv_path,
    struct_root,
    profile_root,
    name_col="name",
    seq_col="aa_seq",
    dg_col="dG_ML",
):
    df = pd.read_csv(csv_path, low_memory=False)

    seqs, folds, structs, profiles, names = [], [], [], [], []
    for name, seq, dg in zip(df[name_col], df[seq_col], df[dg_col]):
        clean_name = rename_data(name).replace(".pdb", "").replace("__", "_")
        dg = float(dg)

        seqs.append(seq)
        folds.append(1 if dg > 0 else 0)
        structs.append(f"{struct_root}/{clean_name}.pt")
        profiles.append(f"{profile_root}/{clean_name}_profile.npy")
        names.append(clean_name)

    return {
        "seqs": seqs,
        "fold": folds,
        "structs": structs,
        "profiles": profiles,
        "names": names,
    }


###############################
# Loading the Mega-scale data #
###############################
def load_train_val(split_num, add_extremes_to_cls=True):
    base = "../processed_data/csv"
    struct_root = "../processed_data/Mega_ProteinMPNN_node_rep"
    profile_root = "../processed_data/Mega_ProteinMPNN_output_profile"

    reg_train = build_regression_dataset_from_csv(
        f"{base}/split_{split_num}/train_numeric_dG.csv",
        struct_root,
        profile_root,
        name_col="name",
        seq_col="aa_seq",
        dg_col="dG_ML",
    )
    reg_val = build_regression_dataset_from_csv(
        f"{base}/split_{split_num}/validation_numeric_dG.csv",
        struct_root,
        profile_root,
        name_col="name",
        seq_col="aa_seq",
        dg_col="dG_ML",
    )

    cls_train = build_classification_dataset_from_csv(
        f"{base}/split_{split_num}/train_numeric_dG.csv",
        struct_root,
        profile_root,
        name_col="name",
        seq_col="aa_seq",
        dg_col="dG_ML",
    )


    if add_extremes_to_cls:
        for fname, pseudo_dg in [
            (f"{base}/split_{split_num}/train_dG_lt_minus1.csv", -1.0),
            (f"{base}/split_{split_num}/train_dG_gt_5.csv", 5.0),
        ]:
            df = pd.read_csv(f"{fname}", low_memory=False)
            for name, seq in zip(df["name"], df["aa_seq"]):
                clean_name = rename_data(name).replace(".pdb", "").replace("__", "_")
                cls_train["seqs"].append(seq)
                cls_train["fold"].append(1 if pseudo_dg > 0 else 0)
                cls_train["structs"].append(f"{struct_root}/{clean_name}.pt")
                cls_train["profiles"].append(f"{profile_root}/{clean_name}_profile.npy")
                cls_train["names"].append(clean_name)


    return reg_train, reg_val, cls_train

def load_mega_test(split_num):
    base = "../processed_data/csv"
    struct_root = "../processed_data/Mega_ProteinMPNN_node_rep"
    profile_root = "../processed_data/Mega_ProteinMPNN_output_profile"

    return build_regression_dataset_from_csv(
        f"{base}/split_{split_num}/test_numeric_dG.csv",
        struct_root,
        profile_root,
        name_col="name",
        seq_col="aa_seq",
        dg_col="dG_ML",
    )



###############################################################
# Data loading for the analysis to make Figure 4c, S3, S4, S5 #
###############################################################
def load_mega_test_and_val_wt_only(split_num):
    base = "../processed_data/csv"
    struct_root = "../processed_data/Mega_ProteinMPNN_node_rep"
    profile_root = "../processed_data/Mega_ProteinMPNN_output_profile"

    csv_path = f"{base}/split_{split_num}/test_numeric_dG.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    if "WT_name" not in df.columns:
        raise KeyError(f"'WT_name' column was not found in {csv_path}")

    seqs, dgs, structs, profiles, names, pdbs = [], [], [], [], [], []

    for name, wt_name, seq, dg in zip(df["name"], df["mmseqs_cluster"], df["aa_seq"], df["dG_ML"]):
        clean_name = rename_data(name).replace(".pdb", "").replace("__", "_")
        clean_wt_name = rename_data(wt_name).replace(".pdb", "").replace("__", "_")

        if clean_name != clean_wt_name:
            continue

        pdb_path = resolve_wt_pdb_path(clean_name)

        seqs.append(seq)
        dgs.append(float(dg))
        structs.append(f"{struct_root}/{clean_name}.pt")
        profiles.append(f"{profile_root}/{clean_name}_profile.npy")
        names.append(clean_name)
        pdbs.append(pdb_path)

    csv_path = f"{base}/split_{split_num}/validation_numeric_dG.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    if "WT_name" not in df.columns:
        raise KeyError(f"'WT_name' column was not found in {csv_path}")

    for name, wt_name, seq, dg in zip(df["name"], df["mmseqs_cluster"], df["aa_seq"], df["dG_ML"]):
        clean_name = rename_data(name).replace(".pdb", "").replace("__", "_")
        clean_wt_name = rename_data(wt_name).replace(".pdb", "").replace("__", "_")

        if clean_name != clean_wt_name:
            continue

        pdb_path = resolve_wt_pdb_path(clean_name)

        seqs.append(seq)
        dgs.append(float(dg))
        structs.append(f"{struct_root}/{clean_name}.pt")
        profiles.append(f"{profile_root}/{clean_name}_profile.npy")
        names.append(clean_name)
        pdbs.append(pdb_path)


    print(f"[INFO] WT-only export dataset: {len(names)} samples loaded from split {split_num}")

    return {
        "seqs": seqs,
        "dgs": dgs,
        "structs": structs,
        "profiles": profiles,
        "names": names,
        "pdbs": pdbs,
    }

############################
# Loading the Maxwell data #
############################
def build_maxwell_dataset():
    df = pd.read_csv("../processed_data/csv/Maxwell_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv", low_memory=False)

    seqs, dgs, structs, profiles, names = [], [], [], [], []
    for protein_id, seq, dg in zip(df["id"], df["sequence"], df["dg"]):
        struct_hits = glob.glob(f"../processed_data/Maxwell_ProteinMPNN_node_rep/{protein_id}*.pt")
        profile_hits = glob.glob(f"../processed_data/Maxwell_ProteinMPNN_output_profile/{protein_id}*.npy")

        if len(struct_hits) == 0:
            raise FileNotFoundError(f"No Maxwell struct .pt found for id={protein_id}")
        if len(profile_hits) == 0:
            raise FileNotFoundError(f"No Maxwell profile .npy found for id={protein_id}")

        dg_kcal = float(dg) * 0.239006

        seqs.append(seq)
        dgs.append(dg_kcal)
        structs.append(struct_hits[0])
        profiles.append(profile_hits[0])
        names.append(str(protein_id))

    return {
        "seqs": seqs,
        "dgs": dgs,
        "structs": structs,
        "profiles": profiles,
        "names": names,
    }

#####################
# Optimizer setting #
#####################
def make_optimizer(model, base_lr=1e-6, head_lr=1e-4, weight_decay=0.01):
    head_keywords = [
        "MLP_pe_",
        "ProteinMPNN_rep_refine",
        "ESM_rep_refine",
        "interaction_proj",
        "aa_res_head",
        "CA_aa_seq_rep_and_node_rep",
    ]

    head_params, encoder_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in head_keywords):
            head_params.append(p)
        else:
            encoder_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]
    )
    return optimizer

####################################################
# Model performance evaluation during the training #
####################################################
def evaluate_regression(
    model,
    dataset,
    batch_size=64,
    print_predictions=False,
    print_prefix="",
    show_progress=True,
    progress_desc="validation",
):
    model.eval()

    pred_dg, true_dg = [], []
    running_loss = 0.0
    n_batches = 0

    batch_iter = iterate_batches(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sample_fraction=1.0,
    )

    if show_progress:
        total_batches = math.ceil(len(dataset["seqs"]) / batch_size)
        batch_iter = tqdm.tqdm(batch_iter, total=total_batches, desc=progress_desc)

    with torch.no_grad():
        for batch in batch_iter:
            out = model(batch["seqs"], batch["structs"], batch["profiles"], return_details=True)
            dg_pred = out["dG"]

            loss = F.huber_loss(dg_pred, batch["dgs"], reduction="mean", delta=1.0)
            running_loss += float(loss.item())
            n_batches += 1

            dg_pred_cpu = dg_pred.detach().cpu().tolist()
            dg_true_cpu = batch["dgs"].detach().cpu().tolist()

            if print_predictions:
                for p_dg, e_dg in zip(dg_pred_cpu, dg_true_cpu):
                    if print_prefix:
                        print(print_prefix, end="")
                    print(round(float(p_dg), 2), round(float(e_dg), 2))

            pred_dg.extend(dg_pred_cpu)
            true_dg.extend(dg_true_cpu)

    pred_dg = np.asarray(pred_dg, dtype=float)
    true_dg = np.asarray(true_dg, dtype=float)

    pearson = safe_corr(pred_dg, true_dg)
    spearman = safe_spearman(pred_dg, true_dg)
    rmse = float(np.sqrt(np.mean((pred_dg - true_dg) ** 2))) if len(pred_dg) > 0 else float("nan")
    mae = float(np.mean(np.abs(pred_dg - true_dg))) if len(pred_dg) > 0 else float("nan")

    return {
        "val_loss": running_loss / max(n_batches, 1),
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "mae": mae,
        "pred_dg": pred_dg.tolist(),
        "true_dg": true_dg.tolist(),
    }


def composite_score(metrics):
    pearson = 0.0 if math.isnan(metrics["pearson"]) else metrics["pearson"]
    spearman = 0.0 if math.isnan(metrics["spearman"]) else metrics["spearman"]
    return pearson + spearman

###############################################################################
# Functions for an epoch training of foldability prediction and ΔG regression #
###############################################################################
def train_one_epoch_classification(
    model,
    dataset,
    optimizer,
    scheduler,
    batch_size,
    epoch_label="cls",
    sample_fraction=1.0,
):
    model.train()
    model.aa_seq_encoder.eval()

    running = 0.0
    n_batches = 0

    sampled_idx = get_sampled_indices(
        n_items=len(dataset["seqs"]),
        sample_fraction=sample_fraction,
        shuffle=True,
    )
    sampled_labels = [dataset["fold"][i] for i in sampled_idx]
    pos_weight, n_pos, n_neg = compute_binary_pos_weight(sampled_labels)

    for start in tqdm.tqdm(range(0, len(sampled_idx), batch_size), desc=epoch_label):
        batch_idx = sampled_idx[start:start + batch_size]

        batch = {
            "seqs": [dataset["seqs"][i] for i in batch_idx],
            "structs": [dataset["structs"][i] for i in batch_idx],
            "profiles": [dataset["profiles"][i] for i in batch_idx],
            "names": [dataset["names"][i] for i in batch_idx],
            "fold": torch.tensor(
                [dataset["fold"][i] for i in batch_idx],
                dtype=torch.float32,
                device=device,
            ),
        }

        out = model(batch["seqs"], batch["structs"], batch["profiles"], return_details=True)
        fold_logit = out["foldability_logit"]

        cls_loss = F.binary_cross_entropy_with_logits(
            fold_logit,
            batch["fold"],
            reduction="mean",
            pos_weight=pos_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        cls_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running += float(cls_loss.item())
        n_batches += 1

    print(
        f"[{epoch_label}] cls_pos_weight={float(pos_weight.item()):.4f} "
        f"(n_pos={n_pos}, n_neg={n_neg})"
    )

    return running / max(n_batches, 1)


def train_one_epoch_regression(
    model,
    dataset,
    optimizer,
    scheduler,
    batch_size,
    epoch_label="reg",
    sample_fraction=1.0,
):
    model.train()
    model.aa_seq_encoder.eval()

    running = 0.0
    n_batches = 0

    for batch in tqdm.tqdm(
        iterate_batches(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            sample_fraction=sample_fraction,
        ),
        desc=epoch_label
    ):
        out = model(batch["seqs"], batch["structs"], batch["profiles"], return_details=True)
        dg_pred = out["dG"]

        reg_loss = F.huber_loss(dg_pred, batch["dgs"], reduction="mean", delta=1.0)

        optimizer.zero_grad(set_to_none=True)
        reg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running += float(reg_loss.item())
        n_batches += 1

    return running / max(n_batches, 1)



###################################################################
# Helper functions for the bootstrap analysis in the Maxwell test #
###################################################################
def _bootstrap_regression_metrics(pred_dg, true_dg, B=10000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    pred_dg = np.asarray(pred_dg, dtype=float)
    true_dg = np.asarray(true_dg, dtype=float)
    N = pred_dg.shape[0]

    metric_names = ["pearson", "spearman", "rmse", "mae", "val_loss"]
    samples = {k: [] for k in metric_names}

    point_metrics = evaluate_regression_from_predictions(pred_dg, true_dg)

    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        boot_metrics = evaluate_regression_from_predictions(pred_dg[idx], true_dg[idx])
        for k in metric_names:
            v = float(boot_metrics[k])
            if v == v:
                samples[k].append(v)

    summary = {}
    for k in metric_names:
        arr = np.asarray(samples[k], dtype=float)
        if arr.size == 0:
            summary[k] = {
                "value": float(point_metrics[k]),
                "ci95": [float("nan"), float("nan")],
                "n_bootstrap_valid": 0,
            }
            continue

        record = {
            "value": float(point_metrics[k]),
            "ci95": [
                float(np.quantile(arr, alpha / 2)),
                float(np.quantile(arr, 1 - alpha / 2)),
            ],
            "n_bootstrap_valid": int(arr.size),
        }

        if k in ("pearson", "spearman"):
            record["p_le_0"] = float((arr <= 0.0).mean())

        summary[k] = record

    summary["bootstrap_config"] = {
        "B": int(B),
        "seed": int(seed),
        "alpha": float(alpha),
        "n_samples": int(N),
    }
    return summary


#############################
# Model ensemble helpers    #
#############################
def resolve_ensemble_ckpts(ensemble_root=DEFAULT_ENSEMBLE_ROOT, splits=DEFAULT_ENSEMBLE_SPLITS):
    ckpt_paths = []
    for split in splits:
        if ensemble_root==DEFAULT_ENSEMBLE_ROOT:
            ckpt = os.path.join(ensemble_root, f"pth_split_{split}", "LPM.pth")
        elif ensemble_root==DEFAULT_ENSEMBLE_ROOT_TRAINED_BY_USER:
            ckpt = os.path.join(ensemble_root, f"pth_split_{split}", "best.pth")
            
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Ensemble checkpoint not found: {ckpt}")
        ckpt_paths.append(ckpt)
    return ckpt_paths


def _copy_non_esm_modules(src_model, dst_model):
    skip_prefix = "aa_seq_encoder."
    src_sd = src_model.state_dict()
    filtered_sd = {k: v for k, v in src_sd.items() if not k.startswith(skip_prefix)}
    missing, unexpected = dst_model.load_state_dict(filtered_sd, strict=False)

    unexpected = [x for x in unexpected if not x.startswith(skip_prefix)]
    missing = [x for x in missing if not x.startswith(skip_prefix)]

    if len(unexpected) > 0:
        raise RuntimeError(f"Unexpected keys while copying non-ESM modules: {unexpected}")
    if len(missing) > 0:
        print(f"[WARN] Missing non-ESM keys while copying modules: {missing}")


def build_shared_ensemble_models(ckpt_paths, ESM_size=ESM_SIZE, dropout=0.1):
    from RINAMI_model_main import RINAMIInterpretableMultiTask

    if len(ckpt_paths) == 0:
        raise ValueError("ckpt_paths is empty")

    print("[INFO] Building ensemble models with shared ESM encoder")

    base_model = RINAMIInterpretableMultiTask(dropout=dropout, ESM_size=ESM_size).to(device)
    base_state = torch.load(ckpt_paths[0], map_location=device)
    base_model.load_state_dict(base_state, strict=False)

    shared_esm = base_model.aa_seq_encoder
    shared_esm.eval()
    for p in shared_esm.parameters():
        p.requires_grad = False

    ensemble_models = []
    for idx, ckpt in enumerate(ckpt_paths):
        model_i = RINAMIInterpretableMultiTask(dropout=dropout, ESM_size=ESM_size).to(device)
        model_i.aa_seq_encoder = shared_esm

        tmp_model = RINAMIInterpretableMultiTask(dropout=dropout, ESM_size=ESM_size).to(device)
        tmp_state = torch.load(ckpt, map_location=device)
        tmp_model.load_state_dict(tmp_state, strict=False)

        _copy_non_esm_modules(tmp_model, model_i)

        model_i.eval()
        ensemble_models.append(model_i)
        print(f"[INFO] Loaded ensemble head {idx+1}/{len(ckpt_paths)} from: {ckpt}")

        del tmp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ensemble_models


def _avg_ensemble_matrix_outputs(ensemble_models, seqs, structs, profiles):
    residue_amino_acid_wise_dG_mats = []

    with torch.no_grad():
        for model in ensemble_models:
            out = model(seqs, structs, profiles, return_details=True)
            residue_amino_acid_wise_dG_mats.append(out["residue_amino_acid_wise_dG_mat"])

    residue_amino_acid_wise_dG_mat_avg = torch.stack(residue_amino_acid_wise_dG_mats, dim=0).mean(dim=0)
    aa_seq_onehots = aa_sequences_to_padded_onehot(seqs).to(device)

    seq_lengths = torch.tensor([len(seq) for seq in seqs], device=device)
    max_len = residue_amino_acid_wise_dG_mat_avg.shape[1]
    valid_mask = torch.arange(max_len, device=device)[None, :] < seq_lengths[:, None]

    residue_amino_acid_wise_dG_mat_avg = residue_amino_acid_wise_dG_mat_avg * valid_mask.unsqueeze(-1).float()
    partial_dG_scores = (residue_amino_acid_wise_dG_mat_avg * aa_seq_onehots).sum(dim=-1)
    partial_dG_scores = partial_dG_scores * valid_mask.float()

    dG = partial_dG_scores.sum(dim=-1)
    foldability_logit = dG
    foldability_prob = torch.sigmoid(foldability_logit)

    return {
        "residue_amino_acid_wise_dG_mat_avg": residue_amino_acid_wise_dG_mat_avg,
        "dG": dG,
        "foldability_logit": foldability_logit,
        "foldability_prob": foldability_prob,
    }

##############################################
# Maxwell test (predict with model ensemble) #
##############################################
def evaluate_maxwell_ensemble_from_avg_matrix(
    ensemble_models,
    dataset,
    batch_size=64,
    print_predictions=False,
    show_progress=True,
    progress_desc="Maxwell ensemble",
):
    if len(ensemble_models) == 0:
        raise ValueError("ensemble_models is empty")

    pred_dg, true_dg = [], []
    pred_fold_logit, pred_fold_prob = [], []

    batch_iter = iterate_batches(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sample_fraction=1.0,
    )

    if show_progress:
        total_batches = math.ceil(len(dataset["seqs"]) / batch_size)
        batch_iter = tqdm.tqdm(batch_iter, total=total_batches, desc=progress_desc)

    printed_header = False

    with torch.no_grad():
        for batch in batch_iter:
            out = _avg_ensemble_matrix_outputs(
                ensemble_models,
                batch["seqs"],
                batch["structs"],
                batch["profiles"],
            )

            dg_pred = out["dG"]
            foldability_logit = out["foldability_logit"]
            foldability_prob = out["foldability_prob"]

            dg_pred_cpu = dg_pred.detach().cpu().tolist()
            dg_true_cpu = batch["dgs"].detach().cpu().tolist()
            fold_logit_cpu = foldability_logit.detach().cpu().tolist()
            fold_prob_cpu = foldability_prob.detach().cpu().tolist()

            if print_predictions and not printed_header:
                print(
                    f"{'name':<24} "
                    f"{'p_dG':>10} "
                    f"{'e_dG':>10} "
                    f"{'fold_logit':>14} "
                    f"{'fold_prob':>12}"
                )
                print("-" * 74)
                printed_header = True

            if print_predictions:
                for name, p_dg, e_dg, flogit, fprob in zip(
                    batch["names"],
                    dg_pred_cpu,
                    dg_true_cpu,
                    fold_logit_cpu,
                    fold_prob_cpu,
                ):
                    print(
                        f"{name:<24.24} "
                        f"{float(p_dg):>10.4f} "
                        f"{float(e_dg):>10.4f} "
                        f"{float(flogit):>14.4f} "
                        f"{float(fprob):>12.4f}"
                    )

            pred_dg.extend(dg_pred_cpu)
            true_dg.extend(dg_true_cpu)
            pred_fold_logit.extend(fold_logit_cpu)
            pred_fold_prob.extend(fold_prob_cpu)

    metrics = evaluate_regression_from_predictions(pred_dg, true_dg)
    metrics["pred_foldability_logit"] = pred_fold_logit
    metrics["pred_foldability_prob"] = pred_fold_prob
    return metrics


##################################################
# Garcia benchmark (predict with model ensemble) #
##################################################
def load_garcia_benchmark_dataset(seq_len_threshold=300):
    header_to_label = {}
    for line in open("../processed_data/fasta/Garcia_benchmark_CD_measured.fasta"):
        if ">" in line:
            label = line.replace(">", "").strip().split("_")[-1]
            header = line.replace(">", "").strip().replace(label, "")[:-1]
            header_to_label[header] = label

    df = pd.read_csv("../processed_data/csv/Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv")

    if "Name" not in df.columns:
        raise KeyError("'Name' column was not found in ../processed_data/csv/Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv")
    if "AlphaFold_pLDDT3recycles" not in df.columns:
        raise KeyError("'AlphaFold_pLDDT3recycles' column was not found in ../processed_data/csv/Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv")
    if "ESMFold_pLDDT" not in df.columns:
        raise KeyError("'ESMFold_pLDDT' column was not found in ../processed_data/csv/Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv")

    benchmark_data_dict = {}
    for name, af_plddt_3rec, esmfold_plddt in zip(
        df["Name"],
        df["AlphaFold_pLDDT3recycles"],
        df["ESMFold_pLDDT"],
    ):
        benchmark_data_dict[name] = {
            "AF_pLDDT_3rec": af_plddt_3rec,
            "ESMFold_pLDDT": esmfold_plddt,
        }

    struct_list_test_data = []
    foldability_label_test_data = []
    mpnn_profile_test_data = []
    seq_list_test_data = []
    names_test_data = []

    AF_pLDDT_3rec_and_label = [[], []]
    ESMFold_pLDDT_and_label = [[], []]

    for pdb in glob.glob("../processed_data/Garcia_benchmark_predicted_structure_pdb/*.pdb"):
        seq = get_sequence_from_single_chain_pdb(pdb)
        if len(seq) > seq_len_threshold:
            continue

        label = pdb.split("/")[-1].replace(".pdb", "").split("_")[-1]
        name = pdb.split("/")[-1].replace(".pdb", "")
        design_name = name.replace("_" + label, "")

        if design_name not in header_to_label:
            continue
        if design_name not in benchmark_data_dict:
            continue

        exp_foldability_label = header_to_label[design_name]
        foldability_label = 1 if exp_foldability_label == "True" else 0

        af_plddt = benchmark_data_dict[design_name]["AF_pLDDT_3rec"]
        esm_plddt = benchmark_data_dict[design_name]["ESMFold_pLDDT"]

        if pd.isna(af_plddt) or pd.isna(esm_plddt):
            continue

        AF_pLDDT_3rec_and_label[0].append(foldability_label)
        AF_pLDDT_3rec_and_label[1].append(float(af_plddt) / 100.0)

        ESMFold_pLDDT_and_label[0].append(foldability_label)
        ESMFold_pLDDT_and_label[1].append(float(esm_plddt) / 100.0)

        foldability_label_test_data.append(foldability_label)
        struct_hits = glob.glob(f"../processed_data/Garcia_benchmark_ProteinMPNN_node_rep/{name}*.pt")
        profile_hits = glob.glob(f"../processed_data/Garcia_benchmark_ProteinMPNN_output_profile/{name}*.npy")

        if len(struct_hits) == 0:
            raise FileNotFoundError(f"No Garcia struct .pt found for {name}")
        if len(profile_hits) == 0:
            raise FileNotFoundError(f"No Garcia profile .npy found for {name}")

        struct_list_test_data.append(struct_hits[0])
        mpnn_profile_test_data.append(profile_hits[0])
        seq_list_test_data.append(seq)
        names_test_data.append(name)

    return {
        "seqs": seq_list_test_data,
        "structs": struct_list_test_data,
        "profiles": mpnn_profile_test_data,
        "fold": foldability_label_test_data,
        "names": names_test_data,
        "af_plddt": AF_pLDDT_3rec_and_label[1],
        "esmfold_plddt": ESMFold_pLDDT_and_label[1],
        "af_label": AF_pLDDT_3rec_and_label[0],
    }


def evaluate_foldability_ensemble_from_avg_matrix(
    ensemble_models,
    dataset,
    batch_size=64,
    threshold=0.5,
    print_predictions=False,
    show_progress=True,
    progress_desc="Garcia ensemble",
):
    if len(ensemble_models) == 0:
        raise ValueError("ensemble_models is empty")

    p_f_probs = []
    p_f_list = []
    e_f_list = []

    batch_iter = iterate_batches(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sample_fraction=1.0,
    )

    if show_progress:
        total_batches = math.ceil(len(dataset["seqs"]) / batch_size)
        batch_iter = tqdm.tqdm(batch_iter, total=total_batches, desc=progress_desc)

    printed_header = False

    with torch.no_grad():
        for batch in batch_iter:
            out = _avg_ensemble_matrix_outputs(
                ensemble_models,
                batch["seqs"],
                batch["structs"],
                batch["profiles"],
            )

            foldability_logit = out["foldability_logit"]
            foldability_prob = out["foldability_prob"]

            probs_cpu = foldability_prob.detach().cpu().tolist()
            labels_cpu = batch["fold"].detach().cpu().tolist()
            logits_cpu = foldability_logit.detach().cpu().tolist()

            if print_predictions and not printed_header:
                print(
                    f"{'name':<32} "
                    f"{'logit':>12} "
                    f"{'prob':>12} "
                    f"{'pred':>8} "
                    f"{'true':>8}"
                )
                print("-" * 80)
                printed_header = True

            for name, prob, label, logit in zip(
                batch["names"], probs_cpu, labels_cpu, logits_cpu
            ):
                pred = 1 if prob > threshold else 0
                p_f_probs.append(float(prob))
                p_f_list.append(pred)
                e_f_list.append(int(label))

                if print_predictions:
                    print(
                        f"{name:<32.32} "
                        f"{float(logit):>12.4f} "
                        f"{float(prob):>12.4f} "
                        f"{int(pred):>8d} "
                        f"{int(label):>8d}"
                    )

    y_true = np.array(e_f_list, dtype=int)
    probs = np.array(p_f_probs, dtype=float)

    try:
        auc_roc = float(roc_auc_score(y_true, probs))
    except ValueError:
        auc_roc = float("nan")

    try:
        auc_pr = float(average_precision_score(y_true, probs))
    except ValueError:
        auc_pr = float("nan")

    acc = float(accuracy_score(e_f_list, p_f_list))
    true_pos_count = int((y_true == 1).sum())
    true_neg_count = int((y_true == 0).sum())

    return {
        "p_f_probs": p_f_probs,
        "p_f_list": p_f_list,
        "e_f_list": e_f_list,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "acc": acc,
        "true_pos_count": true_pos_count,
        "true_neg_count": true_neg_count,
    }


####################################################
# Compute the evaluation metrics for ΔG regression #
####################################################
def evaluate_regression_from_predictions(pred_dg, true_dg):
    pred_dg = np.asarray(pred_dg, dtype=float)
    true_dg = np.asarray(true_dg, dtype=float)

    if len(pred_dg) == 0:
        return {
            "val_loss": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "pred_dg": [],
            "true_dg": [],
        }

    pred_t = torch.tensor(pred_dg, dtype=torch.float32)
    true_t = torch.tensor(true_dg, dtype=torch.float32)
    val_loss = float(F.huber_loss(pred_t, true_t, reduction="mean", delta=1.0).item())

    pearson = safe_corr(pred_dg, true_dg)
    spearman = safe_spearman(pred_dg, true_dg)
    rmse = float(np.sqrt(np.mean((pred_dg - true_dg) ** 2)))
    mae = float(np.mean(np.abs(pred_dg - true_dg)))

    return {
        "val_loss": val_loss,
        "pearson": pearson,
        "spearman": spearman,
        "rmse": rmse,
        "mae": mae,
        "pred_dg": pred_dg.tolist(),
        "true_dg": true_dg.tolist(),
    }
