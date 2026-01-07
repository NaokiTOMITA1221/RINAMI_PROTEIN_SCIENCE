import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import tqdm
import layers
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from util import batch_maker, batch_maker_for_inputs, aa_sequences_to_padded_onehot, pad_feature_matrices
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import json
from torch.nn.utils.rnn import pad_sequence
import glob
import sys
import math
from Bio.PDB import PDBParser, PPBuilder
import random
from typing import List, Tuple, Any, Dict
import gzip
import csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ==============================================================
# PDB から配列抽出（既存）
# ==============================================================

def get_sequence_from_single_chain_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ppb = PPBuilder()

    model = structure[0]
    chain = list(model.get_chains())[0]
    sequence = ppb.build_peptides(chain)[0].get_sequence()
    return str(sequence)


# ==============================================================
# モデル本体（既存・一部軽微な安全化のみ）
# ==============================================================

class RINAMI(nn.Module):
    def __init__(self, device=device, dropout=0.0, ESM_size=320):
        super().__init__()
        self.device = device
        self.dropout_rate = dropout

        self.in_dim            = 128
        self.profile_dim       = 20
        self.pe_dim            = 256
        self.mid_dim           = 128
        self.aa_rep_dim        = ESM_size
        
        self.aa_seq_encoder     = layers.aa_seq2representation(model_size=self.aa_rep_dim) 
        
        print(f"Dropout rate: {dropout}")
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding
        self.pos_enc              = layers.PositionalEncoding(self.pe_dim  )
        self.MLP_pe_node_rep      = layers.MLP(self.pe_dim  , self.mid_dim)
        self.MLP_pe_aa_seq        = layers.MLP(self.pe_dim  , self.mid_dim)

        # Batch Norm
        self.bn_refine_aa   = nn.BatchNorm1d(self.mid_dim)
        self.bn_refine_node = nn.BatchNorm1d(self.mid_dim)
        self.bn_ic1         = nn.BatchNorm1d(self.mid_dim)
        self.bn_ic2         = nn.BatchNorm1d(self.mid_dim)
        self.bn_ic3         = nn.BatchNorm1d(self.mid_dim)

        # Layer Norm
        self.layer_norm_aa_seq_rep   = nn.LayerNorm(self.aa_rep_dim)
        self.layer_norm_node_rep     = nn.LayerNorm(self.in_dim+self.profile_dim)
        self.layer_norm_interaction1 = nn.LayerNorm(self.mid_dim)
        self.layer_norm_interaction2 = nn.LayerNorm(self.mid_dim)
        self.layer_norm_interaction3 = nn.LayerNorm(self.mid_dim)

        # Projections
        self.ProteinMPNN_rep_refine     = layers.MLP(self.in_dim+self.profile_dim, self.mid_dim)
        self.ESM_rep_refine             = layers.MLP(self.aa_rep_dim, self.mid_dim)

        # MultiHeadCrossAttention
        self.CA_aa_seq_rep_and_node_rep = layers.MultiHeadCrossAttention(self.mid_dim, self.mid_dim, heads=20, dim_head=128)

        # Interaction_converter
        self.interaction_converter1 = layers.MLP(self.mid_dim, self.mid_dim)
        self.interaction_converter2 = layers.MLP(self.mid_dim, self.mid_dim)
        self.interaction_converter3 = layers.MLP(self.mid_dim, self.mid_dim)
        self.interaction_converter4 = layers.MLP(self.mid_dim, 20)

        #helper function for batch normalization with mask
        # x: [B, L, D], bn: nn.BatchNorm1d(D), mask: [B, L]
        def _bn_seq(x, bn, mask=None):
            B, L, D = x.shape
            x2 = x.reshape(B*L, D)
            if mask is not None:
                m = mask.reshape(B*L)
                if m.any():
                    x2_valid = bn(x2[m])
                    x2 = x2.clone()
                    x2[m] = x2_valid
                return x2.view(B, L, D)
            else:
                return bn(x2).view(B, L, D)
        self._bn_seq = _bn_seq

    def forward(self, seq_list, feat_path_list, profile_path_list):
        # getting the embedding of the protein aa sequences
        aa_seq_reps, aa_seq_mask = self.aa_seq_encoder(seq_list)
        aa_seq_onehots = aa_sequences_to_padded_onehot(seq_list).to(self.device)
        
        # loading the ProteinMPNN node representation and output profile
        feat_list      = [torch.load(path, weights_only=True)[0] for path in feat_path_list]
        profiles       = [np.load(path).T for path in profile_path_list]
        node_reps      = pad_feature_matrices(feat_list).to(self.device)
        profiles       = pad_feature_matrices(profiles).to(self.device)

        # concat the node representation and output profile
        concated_node_reps   = torch.concat([node_reps, profiles], dim=-1)

        # check the sequence lengthes and make mask
        seq_lengths = torch.tensor([len(seq) for seq in seq_list], device=self.device)
        max_len     = int(seq_lengths.max().item())
        node_mask   = torch.arange(max_len, device=self.device)[None, :] < seq_lengths[:, None]

        # make positional encoding
        pe,_ = layers.create_padded_positional_encodings(self.pos_enc, seq_lengths)

        # refine the structural- and sequence-representations and add PE to refined representations
        refined_aa_seq_reps = self.ESM_rep_refine(self.layer_norm_aa_seq_rep(aa_seq_reps))
        refined_aa_seq_reps = self.dropout(self._bn_seq(refined_aa_seq_reps, self.bn_refine_aa, aa_seq_mask)) + self.MLP_pe_aa_seq(pe.to(self.device))
        refined_node_reps   = self.ProteinMPNN_rep_refine(self.layer_norm_node_rep(concated_node_reps)) + self.MLP_pe_node_rep(pe.to(self.device))
        refined_node_reps   = self.dropout(self._bn_seq(refined_node_reps, self.bn_refine_node, node_mask))

        # CrossAttention
        interaction = self.CA_aa_seq_rep_and_node_rep(
                    refined_aa_seq_reps,
                    refined_node_reps,
                    aa_seq_mask,
                    node_mask
                    )

        # MLPヘッド
        h = self.interaction_converter1(self.layer_norm_interaction1(interaction))
        h = self._bn_seq(h, self.bn_ic1, node_mask); h = F.gelu(h); h = self.dropout(h)
        h = self._bn_seq(self.interaction_converter2(self.layer_norm_interaction2(h)), self.bn_ic2, node_mask); h = F.gelu(h); h = self.dropout(h)
        h = self._bn_seq(self.interaction_converter3(self.layer_norm_interaction3(h)), self.bn_ic3, node_mask); h = F.gelu(h); h = self.dropout(h)
        scores = self.interaction_converter4(h)
        
        mask_to_hadamard = node_mask.unsqueeze(-1)
        hadamard = torch.mul(scores, aa_seq_onehots) * mask_to_hadamard
        
        foldability_logit = hadamard.sum(dim=(1, 2))
        return foldability_logit


# ==============================================================
# 学習（不均衡対策込み）
# ==============================================================

def train_model(model_save_path, trained_model_param=None, num_epochs=5, batch_size=128, dropout=0., pth_ind=None, ESM_size=320):
    decoy_to_seq_dict = json.load(open('../processed_data/decoy_to_seq_dict.json'))
    
    ##########################
    # Loading training data  #
    ##########################
    df_train_data = pd.read_csv('../processed_data/csv/mega_train.csv')
    struct_list_train_data, mpnn_profile_train_data = [], []
    for name, wt_name in zip(df_train_data['name'], df_train_data['WT_name']):
        struct_list_train_data.append(f'../processed_data/Mega_ProteinMPNN_node_rep/{name}.pt')
        mpnn_profile_train_data.append(f'../processed_data/Mega_ProteinMPNN_output_profile/{name}_profile.npy')
    seq_list_train_data = list(df_train_data['aa_seq'])
    dG_list_train_data  = list(df_train_data['dG_ML'])

    print('decoy data loading for training...')
    for (name, wt_name) in tqdm.tqdm(zip(df_train_data['name'], df_train_data['WT_name'])):
        name_  = name.replace('.pdb', '')
        aa_seq = decoy_to_seq_dict['decoy_'+name]
        struct_list_train_data.append(f'../processed_data/Mega_decoy_ProteinMPNN_node_rep/decoy_{name}.pt')
        mpnn_profile_train_data.append(f'../processed_data/Mega_decoy_ProteinMPNN_output_profile/decoy_{name}_profile.npy')
        seq_list_train_data.append(aa_seq)
        dG_list_train_data.append(-1)  # デコイは負扱い（<=0）

    ############################
    # Loading validation data  #
    ############################
    df_val_data = pd.read_csv('../processed_data/csv/mega_val.csv')
    struct_list_val_data, mpnn_profile_val_data = [], []
    print('decoy data loading for validation...')
    for (name, wt_name) in tqdm.tqdm(zip(df_val_data['name'], df_val_data['WT_name'])):
        struct_list_val_data.append(f'../processed_data/Mega_ProteinMPNN_node_rep/{name}.pt')
        mpnn_profile_val_data.append(f'../processed_data/Mega_ProteinMPNN_output_profile/{name}_profile.npy')
    seq_list_val_data = list(df_val_data['aa_seq'])
    dG_list_val_data  = list(df_val_data['dG_ML'])
    for name, wt_name in zip(df_val_data['name'], df_val_data['WT_name']):
        name_  = name.replace('.pdb', '')
        aa_seq = decoy_to_seq_dict['decoy_'+name]
        struct_list_val_data.append(f'../processed_data/Mega_decoy_ProteinMPNN_node_rep/decoy_{name}.pt')
        mpnn_profile_val_data.append(f'../processed_data/Mega_decoy_ProteinMPNN_output_profile/decoy_{name}_profile.npy')
        seq_list_val_data.append(aa_seq)
        dG_list_val_data.append(-1)

    ###################
    # Loading RINAMI  #
    ###################
    if trained_model_param is None:
        model = RINAMI(dropout=dropout, ESM_size=ESM_size).to(device)
    else:
        model = RINAMI(dropout=dropout, ESM_size=ESM_size).to(device)
        model.load_state_dict(torch.load(trained_model_param))

    ####################
    # Label processing #
    ####################
    # foldable=1 (dG>0)
    y_train_foldable = np.array([1 if dg > 0 else 0 for dg in dG_list_train_data], dtype=np.int64)
    n_pos = int((y_train_foldable == 1).sum())
    n_neg = int((y_train_foldable == 0).sum())

    ###########################################################
    # Remapping minority data to positive (for using BCELoss) #
    ###########################################################
    if n_pos < n_neg:
        minority_is_foldable = True
        y_train_loss = y_train_foldable.copy()
        N_minor, N_major = n_pos, n_neg
    else:
        minority_is_foldable = False
        y_train_loss = 1 - y_train_foldable
        N_minor, N_major = n_neg, n_pos

    pos_weight_val = float(N_major / max(N_minor, 1))
    pos_weight_val = min(pos_weight_val, 20.0)  # 過大化防止
    print(f"[INFO] Train class counts (foldable=1): pos={n_pos}, neg={n_neg}  |  minority_is_foldable={minority_is_foldable}")
    print(f"[INFO] Using BCEWithLogitsLoss(pos_weight={pos_weight_val:.3f}) with minority-as-positive mapping.")

    ####################################
    # Loss function for dG regression  #
    ####################################
    criterion_1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    #########################
    # Setting for training  #
    #########################
    head_params, encoder_params = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ["interaction_converter", "MLP_pe_", "ESM_rep_projection", "ProteinMPNN_profile_projection"]):
            head_params.append(p)
        else:
            encoder_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-5, "weight_decay": 0.01},
        {"params": head_params,    "lr": 5e-4, "weight_decay": 0.01},
    ])
    steps_per_epoch = math.ceil(len(seq_list_train_data) / batch_size)
    total_steps     = steps_per_epoch * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    #################
    # Training loop #
    #################
    #Training steps
    seed_value = 123
    for epoch in range(num_epochs):
        if epoch > 0:
            continue

        model.train()
        training_loss = 0.0

        balanced_batches = make_balanced_minibatch_indices(
            labels01=y_train_loss.tolist(),
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            seed=seed_value + epoch
        )

        for idx_batch in tqdm.tqdm(balanced_batches):
            aa_seq_batch, struct_batch, profile_batch, dG_batch_list = gather_batch_by_indices(
                seq_list_train_data, struct_list_train_data, mpnn_profile_train_data, dG_list_train_data, idx_batch
            )
            y_batch_minor = torch.tensor([y_train_loss[i] for i in idx_batch], dtype=torch.float32, device=device)

            logits = model(aa_seq_batch, struct_batch, profile_batch)
            loss   = criterion_1(logits, y_batch_minor)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            training_loss += loss.item()

        loss_avg = training_loss / steps_per_epoch

        #Validation step
        model.eval()
        validation_loss = 0.0
        p_fold_probs, y_fold_true = [], []
        steps_val = max(1, math.ceil(len(seq_list_val_data)/batch_size))
        batch_list_val = batch_maker_for_inputs(
            seq_list_val_data, struct_list_val_data, mpnn_profile_val_data, dG_list_val_data, batch_size
        )
        with torch.no_grad():
            for batch in tqdm.tqdm(batch_list_val):
                aa_seq_batch = batch[0]
                struct_batch = batch[1]
                profile_batch = batch[2]
                dG_batch      = torch.tensor(batch[3], dtype=torch.float32, device=device)

                # 評価は foldable=1
                y_fold_batch = torch.tensor([1.0 if dG.item() > 0 else 0.0 for dG in dG_batch], device=device)
                logits = model(aa_seq_batch, struct_batch, profile_batch)

                # 少数=1 の側での val loss
                if minority_is_foldable:
                    y_minor_batch = y_fold_batch
                    probs_minor   = torch.sigmoid(logits)
                else:
                    y_minor_batch = 1.0 - y_fold_batch
                    probs_minor   = 1.0 - torch.sigmoid(logits)

                loss = criterion_1(logits, y_minor_batch)
                validation_loss += loss.item()

                p_fold_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
                y_fold_true.extend(y_fold_batch.detach().cpu().tolist())

        # PR-AUC（少数=1）
        y_fold_true = np.array(y_fold_true, dtype=int)
        probs = np.array(p_fold_probs, dtype=float)
        y_minor_true = y_fold_true if minority_is_foldable else (1 - y_fold_true)
        p_minor_prob = probs        if minority_is_foldable else (1 - probs)
        auprc_minor  = float(average_precision_score(y_minor_true, p_minor_prob))

        # ROC-AUC（foldable=1）
        try:
            auc_roc = float(roc_auc_score(y_fold_true, probs))
        except ValueError:
            auc_roc = float('nan')



        loss_avg_val = validation_loss / steps_val
        print(f"Epoch [{epoch+1}/{num_epochs}]  TrainLoss: {loss_avg:.6f}  ValLoss: {loss_avg_val:.6f}  "
              f"AUPRC(minority=1): {auprc_minor:.4f}  ROC-AUC(foldable=1): {auc_roc:.4f}  ")

        temp_model_save_path = model_save_path.replace('.pth', f'_{epoch}epoch.pth')
        torch.save(model.state_dict(), temp_model_save_path)

            

    torch.save(model.state_dict(), model_save_path)


# ==============================================================
# 検証／テスト（指標強化版）
# ==============================================================

def test_model_with_Rocklin_benchmark_set(trained_model_param, ESM_size, num_epochs=1, batch_size=1, lr=3e-5, dropout=0., pth_ind=None, threshold = 0.5, seq_len_threshold=300):
    header_to_label = {}
    for line in open('../processed_data/fasta/Garcia_benchmark_CD_measured.fasta'):
        if '>' in line:
            label = line.replace('>', '').strip().split('_')[-1]
            header = line.replace('>', '').strip().replace(label, '')[:-1]
            header_to_label[header] = label

    #loading the training data and the validation data
    struct_list_val_data   = []
    foldability_label_val_data = []
    mpnn_profile_val_data  = []
    seq_list_val_data      = []
        
    ##load csv data
    df = pd.read_csv('../processed_data/csv/Garcia_benchmark.csv')
    benchmark_data_dict = {}
    for name, AF_pLDDT_3rec, AF_pLDDT_25rec, ESMFold_pLDDT, MPNN_score, AF_PAE in zip(df['Name'], df['AlphaFold_pLDDT3recycles'], df['AlphaFold_pLDDT25recycles'], df['ESMFold_pLDDT'], df['MPNN_score'], df['AlphaFold_PAE']):
        benchmark_data_dict[name] = {'AF_pLDDT_3rec':AF_pLDDT_3rec, 'AF_pLDDT_25rec':AF_pLDDT_25rec, 'ESMFold_pLDDT':ESMFold_pLDDT, 'MPNN_score':MPNN_score, 'AF_PAE':AF_PAE}
    #load Rocklin_dG_by_ESM_IF
    Rocklin_dG_by_ESM_IF_dict = json.load(open('../processed_data/likelihood_data/Garcia_benchmark.json'))

    ##0th factor is the foldability-label (: 0 or 1).
    AF_pLDDT_3rec_and_label    = [[], []]
    AF_pLDDT_25rec_and_label   = [[], []]
    AF_PAE_and_label           = [[], []]
    ESMFold_pLDDT_and_label    = [[], []]
    MPNN_score_and_label       = [[], []]
    regression_score_and_label = [[], []]


    for pdb in glob.glob('../processed_data/Garcia_benchmark_predicted_structure_pdb/*.pdb'):
        seq    = get_sequence_from_single_chain_pdb(pdb)
        if len(seq)>seq_len_threshold:
            continue
        label       = pdb.split('/')[-1].replace('.pdb', '').split('_')[-1] ##if I use AF2-structure, -1 should replaced to -10.
        name        = pdb.split('/')[-1].replace('.pdb', '')
        design_name = name.replace('_'+label, '')

        if design_name not in header_to_label.keys(): ##if I use AF2-structure, name should replaced to header.
            continue
        exp_foldability_label = header_to_label[design_name]
        
        if exp_foldability_label == 'True':
            foldability_label = 1
        else:
            foldability_label = 0
        
        AF_pLDDT_3rec_and_label[0].append(foldability_label)
        AF_pLDDT_25rec_and_label[0].append(foldability_label)
        AF_PAE_and_label[0].append(foldability_label)
        ESMFold_pLDDT_and_label[0].append(foldability_label)
        MPNN_score_and_label[0].append(foldability_label)
        regression_score_and_label[0].append(foldability_label)

        AF_pLDDT_3rec_and_label[1].append(benchmark_data_dict[design_name]['AF_pLDDT_3rec'])
        AF_pLDDT_25rec_and_label[1].append(benchmark_data_dict[design_name]['AF_pLDDT_25rec'])
        AF_PAE_and_label[1].append(-1*benchmark_data_dict[design_name]['AF_PAE']) ##-1 is multiplied because PAE is error.
        ESMFold_pLDDT_and_label[1].append(benchmark_data_dict[design_name]['ESMFold_pLDDT'])
        MPNN_score_and_label[1].append(benchmark_data_dict[design_name]['MPNN_score'])

        
        regression_score = Rocklin_dG_by_ESM_IF_dict[name]
        regression_score_and_label[1].append(regression_score)

        foldability_label_val_data.append(foldability_label)
        struct_list_val_data.append(glob.glob(f'../processed_data/Garcia_benchmark_ProteinMPNN_node_rep/{name}*.pt')[0])
        mpnn_profile_val_data.append(glob.glob(f'../processed_data/Garcia_benchmark_ProteinMPNN_output_profile/{name}*.npy')[0])
        seq_list_val_data.append(seq)
    
    model = RINAMI( ESM_size=ESM_size).to(device)
    model.load_state_dict(torch.load(trained_model_param))
    criterion_1 = nn.BCEWithLogitsLoss()

    
    p_f_list, e_f_list, p_f_probs = [], [], []
    model.eval()
    validation_loss = 0.0
    steps_val = max(1, math.ceil(len(seq_list_val_data)/batch_size))
    batch_list_val = batch_maker(seq_list_val_data, struct_list_val_data, mpnn_profile_val_data, foldability_label_val_data, batch_size)

    with torch.no_grad():
        for batch in tqdm.tqdm(batch_list_val):
            
            aa_seq_batch  = batch[0]
            struct_batch  = batch[1]
            profile_batch = batch[2]
            foldability_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)

            foldability = model(aa_seq_batch, struct_batch, profile_batch)

            loss = criterion_1(foldability, foldability_batch)
            validation_loss += loss.item()

            for p_f, e_f in zip(torch.sigmoid(foldability.to('cpu')), foldability_batch.to('cpu')):
                p_f_probs.append(float(p_f))
                p_f_list.append(0 if p_f<=0.5 else 1)
                e_f_list.append(int(e_f))
        

    loss_avg_test = validation_loss / steps_val

    y_fold_true = np.array(e_f_list, dtype=int)
    probs_fold  = np.array(p_f_probs, dtype=float)
    n_pos = int((y_fold_true==1).sum()); n_neg = int((y_fold_true==0).sum())
    minority_is_foldable_val = (n_pos < n_neg)

    try:
        auc_roc = float(roc_auc_score(y_fold_true, probs_fold))
    except ValueError:
        auc_roc = float('nan')

    acc = accuracy_score(e_f_list, p_f_list)
    
    true_pos_count = 0
    true_neg_count = 0
    for ef in e_f_list:
        if ef==1:
            true_pos_count += 1
        else:
            true_neg_count += 1
            
    print(f'True Foldable: {true_pos_count}, True Not Foldable: {true_neg_count}')
    print(f"Test Loss: {loss_avg_test:.6f} | "
          f"ROC-AUC(foldable=1): {auc_roc:.4f} | "
          f"Accuracy: {acc:.4f} | ")

    auc_roc_AF_pLDDT_3rec    = float(roc_auc_score(np.array(AF_pLDDT_3rec_and_label[0], dtype=int), np.array(AF_pLDDT_3rec_and_label[1], dtype=float) ))
    auc_roc_AF_pLDDT_25rec   = float(roc_auc_score(np.array(AF_pLDDT_25rec_and_label[0], dtype=int), np.array(AF_pLDDT_25rec_and_label[1], dtype=float) ))
    auc_roc_AF_PAE           = float(roc_auc_score(np.array(AF_PAE_and_label[0], dtype=int), np.array(AF_PAE_and_label[1], dtype=float) ))
    auc_roc_ESMFold_pLDDT    = float(roc_auc_score(np.array(ESMFold_pLDDT_and_label[0], dtype=int), np.array(ESMFold_pLDDT_and_label[1], dtype=float) ))
    auc_roc_MPNN_score       = float(roc_auc_score(np.array(MPNN_score_and_label[0], dtype=int), np.array(MPNN_score_and_label[1], dtype=float) ))
    auc_roc_regression_score = float(roc_auc_score(np.array(regression_score_and_label[0], dtype=int), np.array(regression_score_and_label[1], dtype=float) ))
    
    return auc_roc, auc_roc_AF_pLDDT_3rec, auc_roc_AF_pLDDT_25rec, auc_roc_AF_PAE, auc_roc_ESMFold_pLDDT, auc_roc_MPNN_score, auc_roc_regression_score, true_pos_count, true_neg_count













if __name__ == "__main__":
   """
    Testing  : python3 RINAMI_foldability_prediction_train_and_test.py test_mode <model param path> <test set: "Mega_test", "Maxwell_test", "Garcia_benchmark">
   """
   args = sys.argv

   if len(args) == 2:
       ESM_dim = 320
       print('Training mode...')
       print('basic training step')
       train_model(args[1], num_epochs=1, dropout=0., ESM_size=ESM_dim)

   elif len(args) == 3:
       ESM_dim = 320
       print('Training mode...')
       print(f'training step')
       train_model(args[1], trained_model_param=args[2], num_epochs=1, dropout=0., ESM_size=ESM_dim)
   
   elif len(args) == 4:
        ESM_dim = 320
        trained_model_path = args[-2]
        test_mode          = args[-1]

        if test_mode == 'Garcia_benchmark':
            print('Test mode: Garcia_benchmark_test')
            seq_len_threshold_list =  [80, 130, 180, 230]
            ROC_AUC_list                  = []
            auc_roc_AF_pLDDT_3rec_list    = []
            auc_roc_AF_pLDDT_25rec_list   = []
            auc_roc_AF_PAE_list           = []
            auc_roc_ESMFold_pLDDT_list    = []
            auc_roc_MPNN_score_list       = []
            auc_roc_regression_score_list = []
            true_pos_num_list             = []
            true_neg_num_list             = []
            for seq_len_threshold in seq_len_threshold_list:
                print(f'***************************************************************************************************************************************************************************************************************************************\nseq_len_threshold = {seq_len_threshold}')
                roc_auc, auc_roc_AF_pLDDT_3rec, auc_roc_AF_pLDDT_25rec, auc_roc_AF_PAE, auc_roc_ESMFold_pLDDT, auc_roc_MPNN_score, auc_roc_regression_score, tpc, tnc = test_model_with_Rocklin_benchmark_set(trained_model_path, ESM_size=ESM_dim, seq_len_threshold=seq_len_threshold)
                ROC_AUC_list.append(roc_auc)
                auc_roc_AF_pLDDT_3rec_list.append(auc_roc_AF_pLDDT_3rec)
                auc_roc_AF_pLDDT_25rec_list.append(auc_roc_AF_pLDDT_25rec)
                auc_roc_AF_PAE_list.append(auc_roc_AF_PAE)
                auc_roc_ESMFold_pLDDT_list.append(auc_roc_ESMFold_pLDDT)
                auc_roc_MPNN_score_list.append(auc_roc_MPNN_score)
                auc_roc_regression_score_list.append(auc_roc_regression_score)
                true_pos_num_list.append(tpc)
                true_neg_num_list.append(tnc)
            
            plt.figure(figsize=(8, 5))
            plt.ylim(0.3,0.9)
            plt.plot(seq_len_threshold_list, ROC_AUC_list, c='#f6adc6', linestyle='-', marker='o')
            plt.plot(seq_len_threshold_list, auc_roc_AF_pLDDT_3rec_list, c="gray", linestyle='--', marker='o', alpha=0.5)

            plt.legend(['Our model', 'AlphaFold (3 recycle) pLDDT'])

            plt.xlabel('Sequence length threshold\n (Foldable :  Not Foldable)', fontsize=12, fontweight='bold')
            plt.ylabel('ROC-AUC', fontsize=12, fontweight='bold')
            plt.xticks(seq_len_threshold_list, [f'{seq_len_threshold}\n ({tpc} : {tnc})' for seq_len_threshold, tpc, tnc in zip(seq_len_threshold_list, true_pos_num_list, true_neg_num_list)])

            sns.despine()
            plt.savefig('Trained_model_Garcia_benchmark_result.png', bbox_inches='tight', dpi=300)
            plt.savefig('Trained_model_Garcia_benchmark_result.pdf', bbox_inches='tight', dpi=300)
