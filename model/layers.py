import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import esm
import glob
import os
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import GCNConv, NNConv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################
# pos. enc. for each node from GNN #
####################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, residue_indices):
        return self.pe[residue_indices]

def create_padded_positional_encodings(pe: PositionalEncoding, lengths: list[int]):
    """
    Args:
        pe: PositionalEncoding インスタンス
        lengths: 各配列の長さのリスト [L1, L2, ..., Ln]

    Returns:
        padded_tensor: (batch_size, max_len, d_model)
        mask: (batch_size, max_len)  # True for real positions, False for padding
    """
    batch_encodings = []
    mask_list = []
    max_len = max(lengths)

    for length in lengths:
        indices = torch.arange(length)
        enc = pe(indices)  # (length, d_model)
        pad_len = max_len - length
        padded_enc = torch.cat([enc, torch.zeros(pad_len, enc.size(1), device=enc.device)], dim=0)
        mask = torch.cat([torch.ones(length, dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])
        batch_encodings.append(padded_enc)
        mask_list.append(mask)

    padded_tensor = torch.stack(batch_encodings)  # (batch, max_len, d_model)
    mask = torch.stack(mask_list)  # (batch, max_len)

    return padded_tensor, mask

############################################
# Encoding PDB with pretrained proteinMPNN #
############################################
class PDBGraphEncoder(nn.Module):
    def __init__(self, emb_dim=128, max_len=1000, device=device):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.pos_encoder = PositionalEncoding(emb_dim, max_len)

    def forward(self, proteinMPNN_emb_path_list):
        mpnn_embed_tensors = []

        for path in proteinMPNN_emb_path_list:
            emb = torch.load(path, map_location='cpu', weights_only=False)
            emb = emb + self.pos_encoder(torch.tensor(range(emb.shape[-2]))).to('cpu')
            mpnn_embed_tensors.append(emb)
            
        # パディングして [batch, max_seq_len, emb_dim] のテンソルにする
        padded_batch = pad_sequence(mpnn_embed_tensors, batch_first=True)

        # マスク（パディングされていない部分をTrueに）
        mask = torch.zeros(padded_batch.shape[:2], dtype=torch.bool)
        for i, tensor in enumerate(mpnn_embed_tensors):
            mask[i, :tensor.size(0)] = True


        return padded_batch.to(self.device), mask.to(self.device)



##################################################
# Encoding aa sequences by ESM pre-trained model #
##################################################
class aa_seq2representation(nn.Module):
    
    def __init__(self, device=device, model_size=320):
        super().__init__()
        self.model_dict = {
                        5120: (esm.pretrained.esm2_t48_15B_UR50D, 48),
                        2560: (esm.pretrained.esm2_t36_3B_UR50D, 36),
                        1280: (esm.pretrained.esm2_t33_650M_UR50D, 33),
                        640: (esm.pretrained.esm2_t30_150M_UR50D, 30),
                        480: (esm.pretrained.esm2_t12_35M_UR50D, 12),
                        320: (esm.pretrained.esm2_t6_8M_UR50D, 6),
                     }

        self.device = device
        self.loader_fn, self.n_layers = self.model_dict[model_size] 
        self.lm, self.alphabet = self.loader_fn()
        self.lm.to('cpu')
        self.lm.eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        for name, param in self.lm.named_parameters():
            param.requires_grad = True  ##original: True
    def forward(self, seq_list):
        '''
        input : アミノ酸配列のlist
        output:
            vector_outputs: [batch_size, max_seq_len, embed_dim] のテンソル
            attention_masks: [batch_size, max_seq_len] のアテンションマスク (0: PAD, 1: 実トークン)
        '''
        last_layer_ind = self.n_layers  # モデルによって変更する
        
        len_list = [len(seq) for seq in seq_list]
        max_len = max(len_list)
        seq_input = [(index, seq) for index, seq in enumerate(seq_list)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(seq_input)
        
        with torch.no_grad():
            results = self.lm(batch_tokens.to(self.device), repr_layers=[last_layer_ind], return_contacts=False)

        embed_dim = results["representations"][last_layer_ind].shape[-1]
        batch_size = len(seq_list)

        # パディング用のゼロテンソルを作成
        vector_outputs = torch.zeros((batch_size, max_len, embed_dim))
        attention_masks = torch.zeros((batch_size, max_len), dtype=torch.int32)

        for i, (rep, length) in enumerate(zip(results["representations"][last_layer_ind], len_list)):
            rep_cpu = rep[1:length+1, :].cpu()
            vector_outputs[i, :length, :] = rep_cpu
            attention_masks[i, :length] = 1
        
        del batch_tokens, results
        torch.cuda.empty_cache()
        
        vector_outputs = vector_outputs.to(self.device)
        attention_masks = attention_masks.to(self.device)

        return vector_outputs, attention_masks



####################################################################
# Cross attention for processing Graph datas and sequene embedding #
####################################################################
class MultiHeadCrossAttention(nn.Module):
    """
    drug: (batch_size, drug_len, drug_dim)
    target: (batch_size, target_len, target_dim)
    calculate attention score between drug and target
    """

    def __init__(self, drug_dim=768, target_dim=2560, heads=12, dim_head=128):
        super().__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(drug_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(target_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(target_dim, heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, target_dim)
        
        self.layer_norm = nn.LayerNorm(target_dim)

    def forward(self, drug, target, drug_mask, pro_mask, attn_map_out=False):
        b, n, _, h = *drug.shape, self.heads
        
        # Project drug into query space
        q = self.to_q(drug).view(b, n, self.heads, -1).transpose(1, 2)
        
        # Project target into key and value space
        target_len = target.shape[1]
        k = self.to_k(target).view(b, target_len, self.heads, -1).transpose(1, 2)
        v = self.to_v(target).view(b, target_len, self.heads, -1).transpose(1, 2)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        drug_mask = drug_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.heads, n, target_len)
        masked_dots = dots.masked_fill(drug_mask == 0, -1e6)
        pro_mask = pro_mask.unsqueeze(1).unsqueeze(-2).expand(-1, self.heads, n, target_len)
        masked_dots = masked_dots.masked_fill(pro_mask == 0, -1e6)
        
        # Apply softmax to compute attention weights
        attn = F.softmax(masked_dots, dim=-1)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)

        out = self.to_out(out) + target
        
        ##out = self.layer_norm(out) 
            
        if attn_map_out:
            return out, attn
        else:
            attn.to('cpu')
            del attn
            return out




##########################
# template of MLP layer  #
##########################
class MLP(nn.Module):
    """_summary_

    Args:
    - emb_dim: int, the dimension of the intermediate embeddings
    - num_classes: int, the number of classes to predict
    - dropout: float, the dropout rate

    Returns:
    - z: tensor, the output of the network
    """

    def __init__(self, emb_dim, num_classes, dropout=0.0):

        super().__init__()
        self.desc_skip_connection = True
        #print('dropout is {}'.format(dropout))

        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(emb_dim, emb_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.act2 = nn.GELU()
        self.final = nn.Linear(emb_dim, num_classes)

    def forward(self, inter_emb):
        x_out = self.fc1(inter_emb)
        x_out = self.dropout1(x_out)
        x_out = self.act1(x_out)

        x_out = x_out + inter_emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.act2(z)
        z = self.final(z + x_out)

        return z



##############################################
# shuffle and batchfying the training datas  #
##############################################
def batch_maker(aa___seq_list, base_seq_list, dG_list, batch_size=10, random_shuffle=True):
    zip_list   = list(zip(aa___seq_list, base_seq_list, dG_list))
    batch_list = []
    if random_shuffle:
        random.shuffle(zip_list)
    for batch_ind in range(int(len(aa___seq_list)/batch_size)):
        aa___seq_batch = []
        base_seq_batch = []
        dG_batch = []
        for data in zip_list[batch_ind*batch_size:(batch_ind+1)*batch_size]:
            aa___seq, base_seq, dG_data = data[0], data[1], data[2]
            aa___seq_batch.append(aa___seq)
            base_seq_batch.append(base_seq)
            dG_batch.append(dG_data)
        batch_list.append([aa___seq_batch, base_seq_batch, dG_batch])
    return batch_list

def sinusoidal_positional_encoding(input_array: np.ndarray) -> np.ndarray:
    seq_len, dim = input_array.shape

    # 位置の配列: shape = (seq_len, 1)
    position = np.arange(seq_len)[:, np.newaxis]
    # 周波数スケール: shape = (1, dim)
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))

    # PEの初期化
    pe = np.zeros((seq_len, dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe
                                  

#########################
# GCN layer             #
#########################

class GCN(nn.Module):
    def __init__(self, n_in, n_h, n_out, dropout=0.1):
        super().__init__()
        self.n_in = n_in
        self.n_h  = n_h
        self.n_out = n_out

        self.norm1 = nn.LayerNorm(self.n_in)
        self.norm2 = nn.LayerNorm(self.n_h)
        self.conv1 = GCNConv(self.n_h, self.n_h)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = GCNConv(self.n_h, self.n_h) 
        self.dropout2 = nn.Dropout(dropout)
        self.mlp1 = MLPNet(self.n_in, self.n_h, dropout=dropout)
        self.mlp_skip = MLPNet(self.n_in, self.n_out, dropout=dropout)
        self.mlp2 = MLPNet(self.n_h, self.n_out, dropout=dropout)
        self.relu = nn.GELU()

    def forward(self, data):
        x = self.norm1(data.x)
        x_copy = data.x
        edge_index = data.edge_index
        
        x = self.mlp1(x)
        x = self.conv1(x, edge_index)
        x = self.dropout1(x)
        x = self.relu(x) 

        y = self.conv2(x, edge_index) 
        y = self.dropout2(y)
        y = self.norm2(self.relu(y))
        y = self.mlp2(y)

        z = self.mlp_skip(x_copy) + y
        
        return z


class NNconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_attr_dim = 128
        self.node_in_dim   = 128
        self.hidden_dim    = 128
        self.out_dim       = 128

        self.edge_net = MLPNet(128, self.node_in_dim*self.out_dim)
        self.conv = NNConv(self.node_in_dim, self.out_dim, self.edge_net, aggr='mean')
        self.act = nn.GELU()

    def forward(self, x, edge_index, edge_attr):
        x_copy = x
        x = self.conv(x, edge_index, edge_attr)
        x = x_copy + self.act(x)
        return x
