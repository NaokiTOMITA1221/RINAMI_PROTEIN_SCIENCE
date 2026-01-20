import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.PDB import PDBParser, PPBuilder
import numpy as np
import random
from typing import List, Tuple, Any, Dict
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, accuracy_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################################
# extract AAseq from the A-chain in the input pdb  #
####################################################
def get_sequence_from_single_chain_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ppb = PPBuilder()
    model = structure[0]
    chain = list(model.get_chains())[0]
    sequence = ppb.build_peptides(chain)[0].get_sequence()
    
    return str(sequence)





##############################################
# shuffle and batchfying the training datas  #
##############################################
def batch_maker_for_inputs(aa_seq_list, struct_list, profile_list, dG_list, batch_size=1, random_shuffle=True):
    zip_list   = list(zip(aa_seq_list, struct_list, profile_list, dG_list))
    batch_list = []
    if random_shuffle:
        random.shuffle(zip_list)
    for batch_ind in range(int(len(aa_seq_list)/batch_size)):
        aa_seq_batch        = []
        struct_seq_batch    = []
        rosetta_score_batch = []
        profile_batch       = []
        dG_batch            = []
        for data in zip_list[batch_ind*batch_size:(batch_ind+1)*batch_size]:
            aa_seq, struct_seq, profile_data, dG_data = data[0], data[1], data[2], data[3]
            aa_seq_batch.append(aa_seq)
            struct_seq_batch.append(struct_seq)
            profile_batch.append(profile_data)
            dG_batch.append(dG_data)
        batch_list.append([aa_seq_batch, struct_seq_batch, profile_batch, dG_batch])
    return batch_list





#####################################################
# Converting AA-seqs into the padded one-hot tensor #
#####################################################
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
            # else: すでに padding_value が入っているので無視
    return torch.from_numpy(np.asarray(onehot_tensor, dtype=np.float32))


######################################################################################################
# Organize the list of features (shape of each feature = (seq_len, feature_dim)) into a padded batch #
######################################################################################################
def pad_feature_matrices(feature_list, padding_value=0.0):
    """
    shape = (seq_len, feature_dim) 

    Args:
        features             : List of features (np.ndarray)
        padding_value (float): default = 0.0

    Returns:
        np.ndarray: batch tensor (shape: (batch_size, max_seq_len, feature_dim)) 
    """
    batch_size = len(feature_list)
    max_len = max(mat.shape[0] for mat in feature_list)
    feat_dim = feature_list[0].shape[1]

    padded_tensor = np.full((batch_size, max_len, feat_dim), padding_value, dtype=np.float32)

    for i, mat in enumerate(feature_list):
        seq_len = mat.shape[0]
        padded_tensor[i, :seq_len, :] = mat

    return torch.from_numpy(padded_tensor)





# ==============================================================
# 少数クラス対策：正例(=dG>0)が多い場合は非折りたたみ側を陽性に再マップ
# BCEWithLogitsLoss(pos_weight) と 1:1 近いミニバッチを供給するユーティリティ
# ==============================================================

def make_balanced_minibatch_indices(labels01, batch_size, steps_per_epoch, seed=42):
    """
    labels01: 少数側を1にした 0/1 ラベル配列
    各バッチが概ね 1:1 になるよう、置換抽出で index バッチを返す。
    """
    rng = random.Random(seed)
    idx_minor = [i for i, y in enumerate(labels01) if y == 1]
    idx_major = [i for i, y in enumerate(labels01) if y == 0]

    if len(idx_minor) == 0 or len(idx_major) == 0:
        # 片側しかない場合は通常シャッフル
        all_idx = list(range(len(labels01)))
        rng.shuffle(all_idx)
        return [all_idx[i:i+batch_size] for i in range(0, len(all_idx), batch_size)]

    half1 = batch_size // 2
    half2 = batch_size - half1
    batches = []
    for _ in range(steps_per_epoch):
        b = rng.choices(idx_minor, k=half1) + rng.choices(idx_major, k=half2)
        rng.shuffle(b)
        batches.append(b)
    return batches


def gather_batch_by_indices(seq_list, struct_list, profile_list, dG_list, indices):
    """index リストから各入力を抽出（学習で使用）"""
    aa_seq_batch  = [seq_list[i]     for i in indices]
    struct_batch  = [struct_list[i]  for i in indices]
    profile_batch = [profile_list[i] for i in indices]
    dG_batch      = [dG_list[i]      for i in indices]
    return aa_seq_batch, struct_batch, profile_batch, dG_batch


# ==============================================================
# dG>0 のみを間引いて「正=負」にそろえる（既存の要件通り）
# ==============================================================

def undersample_pos_to_match_neg(
    struct_list_val_data: List[Any],
    mpnn_profile_val_data: List[Any],
    rosetta_score_val_data: List[Any],
    seq_list_val_data: List[Any],
    dG_list_val_data: List[float],
    seed: int = 42,
    keep_zero: bool = True,
) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[float], Dict[str, int]]:
    """
    dG>0（正）だけをランダムに欠落させ、正の件数を負（dG<0）の件数に一致させる。
    dG==0 は既定では保持（クラス数に加算しない）。
    """
    assert len(struct_list_val_data) == len(mpnn_profile_val_data) == len(rosetta_score_val_data) == len(seq_list_val_data) == len(dG_list_val_data), \
        "全リストの長さは等しい必要があります"

    n = len(dG_list_val_data)
    pos_idx = [i for i, dg in enumerate(dG_list_val_data) if dg > 0]
    neg_idx = [i for i, dg in enumerate(dG_list_val_data) if dg < 0]
    zero_idx = [i for i, dg in enumerate(dG_list_val_data) if dg == 0]

    n_pos, n_neg, n_zero = len(pos_idx), len(neg_idx), len(zero_idx)

    drop_count = max(0, n_pos - n_neg)
    random.seed(seed)
    drop_pos = set(random.sample(pos_idx, drop_count)) if drop_count > 0 else set()

    keep_set = set(range(n)) - drop_pos
    if not keep_zero:
        pass

    struct_f = [struct_list_val_data[i]   for i in range(n) if i in keep_set]
    mpnn_f   = [mpnn_profile_val_data[i]  for i in range(n) if i in keep_set]
    ros_f    = [rosetta_score_val_data[i] for i in range(n) if i in keep_set]
    seq_f    = [seq_list_val_data[i]      for i in range(n) if i in keep_set]
    dG_f     = [dG_list_val_data[i]       for i in range(n) if i in keep_set]

    pos_after = sum(1 for x in dG_f if x > 0)
    neg_after = sum(1 for x in dG_f if x < 0)
    zero_after = sum(1 for x in dG_f if x == 0)

    stats = {
        "before_pos": n_pos, "before_neg": n_neg, "before_zero": n_zero, "before_total": n,
        "dropped_pos": drop_count,
        "after_pos": pos_after, "after_neg": neg_after, "after_zero": zero_after, "after_total": len(dG_f),
        "balanced": (pos_after == neg_after)
    }
    return struct_f, mpnn_f, ros_f, seq_f, dG_f, stats



#################################
# helper function for bootstrap #
#################################
def _bootstrap_auc_ci(y_true, y_score, B=10000, seed=0, alpha=0.05):
    """
    y_true: (N,) 0/1
    y_score:(N,) continuous score
    returns: (auc_hat, (ci_low, ci_high), auc_samples, p_auc_le_0p5)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    N = y_true.shape[0]

    # point estimate
    auc_hat = float(roc_auc_score(y_true, y_score))

    auc_samples = []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)  # resample with replacement
        y_b = y_true[idx]
        s_b = y_score[idx]
        # bootstrap sample may have only one class -> AUC undefined
        if len(np.unique(y_b)) < 2:
            continue
        auc_samples.append(float(roc_auc_score(y_b, s_b)))

    auc_samples = np.array(auc_samples, dtype=float)
    if auc_samples.size == 0:
        return auc_hat, (float('nan'), float('nan')), auc_samples, float('nan')

    ci_low = float(np.quantile(auc_samples, alpha/2))
    ci_high = float(np.quantile(auc_samples, 1 - alpha/2))

    # "individual bootstrap test": how often AUC <= 0.5 under bootstrap distribution
    # (interpretation: if this is small, model is very likely > random)
    p_auc_le_0p5 = float((auc_samples <= 0.5).mean())

    return auc_hat, (ci_low, ci_high), auc_samples, p_auc_le_0p5

def _paired_bootstrap_auc_diff_ci(y_true, score_a, score_b, B=10000, seed=0, alpha=0.05):
    """
    paired bootstrap for AUC difference: AUC(score_a) - AUC(score_b)
    returns: (diff_hat, (ci_low, ci_high), diff_samples, p_diff_le_0)
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    N = y_true.shape[0]

    diff_hat = float(roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b))

    diff_samples = []
    for _ in range(B):
        idx = rng.integers(0, N, size=N)
        y_b = y_true[idx]
        a_b = score_a[idx]
        b_b = score_b[idx]
        if len(np.unique(y_b)) < 2:
            continue
        da = roc_auc_score(y_b, a_b)
        db = roc_auc_score(y_b, b_b)
        diff_samples.append(float(da - db))

    diff_samples = np.array(diff_samples, dtype=float)
    if diff_samples.size == 0:
        return diff_hat, (float('nan'), float('nan')), diff_samples, float('nan')

    ci_low = float(np.quantile(diff_samples, alpha/2))
    ci_high = float(np.quantile(diff_samples, 1 - alpha/2))

    # probability that diff <= 0 in bootstrap distribution
    p_diff_le_0 = float((diff_samples <= 0.0).mean())

    return diff_hat, (ci_low, ci_high), diff_samples, p_diff_le_0



