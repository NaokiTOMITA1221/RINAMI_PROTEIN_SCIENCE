import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from Bio.PDB import PDBParser, PPBuilder
import numpy as np
import random


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
            aa_seq, struct_seq, rosetta_score, profile_data, dG_data = data[0], data[1], data[2], data[3], data[4]
            aa_seq_batch.append(aa_seq)
            struct_seq_batch.append(struct_seq)
            rosetta_score_batch.append(rosetta_score)
            profile_batch.append(profile_data)
            dG_batch.append(dG_data)
        batch_list.append([aa_seq_batch, struct_seq_batch, rosetta_score_batch, profile_batch, dG_batch])
    return batch_list



def batch_maker(aa_____seq_list, struct_list, rosetta_score_list, dG_list, batch_size=10, random_shuffle=True):
    zip_list   = list(zip(aa_____seq_list, struct_list, rosetta_score_list, dG_list))
    batch_list = []
    if random_shuffle:
        random.shuffle(zip_list)
    for batch_ind in range(int(len(aa_____seq_list)/batch_size)):
        aa_____seq_batch    = []
        struct_seq_batch    = []
        rosetta_score_batch = []
        dG_batch            = []
        for data in zip_list[batch_ind*batch_size:(batch_ind+1)*batch_size]:
            aa_____seq, struct_seq, rosetta_score, dG_data = data[0], data[1], data[2], data[3]
            aa_____seq_batch.append(aa_____seq)
            struct_seq_batch.append(struct_seq)
            rosetta_score_batch.append(rosetta_score)
            dG_batch.append(dG_data)
        batch_list.append([aa_____seq_batch, struct_seq_batch, rosetta_score_batch, dG_batch])
    return batch_list




def aa_sequences_to_padded_onehot(sequences, aa_order="ACDEFGHIKLMNPQRSTVWY", padding_value=0.0):
    """
    アミノ酸配列リストを one-hot ベクトル列に変換し、最大長にパディングする関数。

    Args:
        sequences (List[str]): アミノ酸配列のリスト
        aa_order (str): one-hot ベクトルで使うアミノ酸の順序（デフォルトで20種）
        padding_value (float): パディング部分に使う値（通常0.0）

    Returns:
        np.ndarray: shape = (N, max_len, 20) の one-hot テンソル
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



def pad_feature_matrices(feature_list, padding_value=0.0):
    """
    shape = [seq_len, feature_dim] の行列リストをパディングしてバッチ化する関数。

    Args:
        feature_list (List[np.ndarray]): 各配列が [seq_len, feature_dim] の行列
        padding_value (float): パディングに使う値（デフォルト: 0.0）

    Returns:
        np.ndarray: shape = (batch_size, max_seq_len, feature_dim) のバッチテンソル
    """
    batch_size = len(feature_list)
    max_len = max(mat.shape[0] for mat in feature_list)
    feat_dim = feature_list[0].shape[1]

    padded_tensor = np.full((batch_size, max_len, feat_dim), padding_value, dtype=np.float32)

    for i, mat in enumerate(feature_list):
        seq_len = mat.shape[0]
        padded_tensor[i, :seq_len, :] = mat

    return torch.from_numpy(padded_tensor)





