import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import esm
import glob
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################################
# Pos. enc. for each representation #
#####################################
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
        pe: PositionalEncoding ()
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
        input : list of aa-sequences
        output:
            vector_outputs: tensor -> shape: [batch_size, max_seq_len, embed_dim] 
            attention_masks: attention_mask_tensor -> shape: [batch_size, max_seq_len] (0: PAD, 1: valid token)
        '''
        last_layer_ind = self.n_layers  # Output layer of the ESM
        
        len_list = [len(seq) for seq in seq_list]
        max_len = max(len_list)
        seq_input = [(index, seq) for index, seq in enumerate(seq_list)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(seq_input)
        
        with torch.no_grad():
            results = self.lm(batch_tokens.to(self.device), repr_layers=[last_layer_ind], return_contacts=False)

        embed_dim = results["representations"][last_layer_ind].shape[-1]
        batch_size = len(seq_list)

        # Making a zero tensor for using in a padding process
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
    Args:
    - sequence representation : (batch_size, seq_len, seq_dim)
    - structure representation: (batch_size, struct_len, struct_dim)
    Returns:
    - reweighted structure representation: (batch_size, struct_len, struct_dim)
    """

    def __init__(self, seq_dim=128, struct_dim=128, heads=20, dim_head=128):
        super().__init__()
        self.seq_dim = seq_dim
        self.struct_dim = struct_dim
        self.heads = heads
        self.scale = dim_head ** -(1/2)

        self.projection_q = nn.Linear(seq_dim, heads * dim_head, bias=False)
        self.projection_k = nn.Linear(struct_dim, heads * dim_head, bias=False)
        self.projection_v = nn.Linear(struct_dim, heads * dim_head, bias=False)
        self.linear_out   = nn.Linear(heads * dim_head, struct_dim)
        

    def forward(self, sequence, structure, seq_mask, struct_mask, attn_map_out=False):
        b, _, _, h = *sequence.shape, self.heads
        seq_len = sequence.shape[1]
        
        # Project sequence into query space
        q = self.projection_q(sequence).view(b, seq_len, self.heads, -1).transpose(1, 2)
        
        # Project structure into key and value space
        struct_len = structure.shape[1]
        k = self.projection_k(structure).view(b, struct_len, self.heads, -1).transpose(1, 2)
        v = self.projection_v(structure).view(b, struct_len, self.heads, -1).transpose(1, 2)
        
        # Compute attention weights
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        seq_mask = seq_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.heads, seq_len, struct_len)
        masked_dots = dots.masked_fill(seq_mask == 0, -1e6)
        struct_mask = struct_mask.unsqueeze(1).unsqueeze(-2).expand(-1, self.heads, seq_len, struct_len)
        masked_dots = masked_dots.masked_fill(struct_mask == 0, -1e6)
        
        # Apply softmax to compute attention weights
        attn = F.softmax(masked_dots, dim=-1)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, seq_len, -1)

        #skip connection
        out = self.linear_out(out) + structure
            
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

    def __init__(self, emb_dim, out_dim, dropout=0.0):

        super().__init__()
        self.desc_skip_connection = True

        self.linear_1     = nn.Linear(emb_dim, emb_dim)
        self.dropout_1    = nn.Dropout(dropout)
        self.act_1        = nn.GELU()
        self.linear_2     = nn.Linear(emb_dim, emb_dim)
        self.dropout_2    = nn.Dropout(dropout)
        self.act_2        = nn.GELU()
        self.final_linear = nn.Linear(emb_dim, out_dim)

    def forward(self, init_emb):
        x_out = self.linear_1(init_emb)
        x_out = self.dropout_1(x_out)
        x_out = self.act_1(x_out)

        x_out = x_out + init_emb

        z = self.linear_2(x_out)
        z = self.dropout_2(z)
        z = self.act_2(z)
        z = self.final_linear(z + x_out)

        return z




                                  

