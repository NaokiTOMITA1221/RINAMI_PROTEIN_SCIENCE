import torch
import torch.nn as nn
import numpy as np

import modules
from util import aa_sequences_to_padded_onehot, pad_feature_matrices


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class BaselineRINAMI(nn.Module):

    def __init__(
        self,
        device=device,
        dropout=0.1,
        ESM_size=320,
    ):
        super().__init__()
        self.device = device
        self.dropout_rate = dropout

        self.in_dim = 128
        self.profile_dim = 20
        self.pe_dim = 128
        self.mid_dim = 128
        self.aa_rep_dim = ESM_size

        # Positional encoding.
        self.pos_enc = modules.PositionalEncoding(self.pe_dim)
        
        # Positional encoding converters.
        self.MLP_pe_node_rep = modules.MLP(self.pe_dim, self.mid_dim, dropout=self.dropout_rate)
        self.MLP_pe_aa_seq = modules.MLP(self.pe_dim, self.mid_dim, dropout=self.dropout_rate)
        self.MLP_pe_interact_1 = modules.MLP(self.pe_dim, self.mid_dim, dropout=self.dropout_rate)
        self.MLP_pe_interact_3 = modules.MLP(self.pe_dim, self.mid_dim, dropout=self.dropout_rate)
        
        # Layer normalization.
        self.layer_norm_node_rep      = nn.LayerNorm(self.in_dim + self.profile_dim)
        self.layer_norm_node_refined_rep   = nn.LayerNorm(self.mid_dim)
        self.layer_norm_interaction_1 = nn.LayerNorm(self.mid_dim)
        self.layer_norm_interaction_2 = nn.LayerNorm(self.mid_dim)
        self.layer_norm_interaction_3 = nn.LayerNorm(self.mid_dim)
        self.layer_norm_interaction_4 = nn.LayerNorm(self.mid_dim)
        self.layer_norm_residue_amino_acid_wise_dG_mat    = nn.LayerNorm(self.mid_dim)
        
        # Projections.
        self.ProteinMPNN_rep_refine = modules.MLP(self.in_dim + self.profile_dim, self.mid_dim)
        
        self.MLP_instead_of_CA = modules.MLP( self.mid_dim, self.mid_dim )
        
        # Interaction representation to residue-amino-acid-wise dG matrix.
        self.interaction_proj_1 = modules.MLP(self.mid_dim, self.mid_dim, dropout=self.dropout_rate)
        self.interaction_proj_2 = modules.MLP(self.mid_dim, self.mid_dim, dropout=self.dropout_rate)
        self.interaction_proj_3 = modules.MLP(self.mid_dim, self.mid_dim, dropout=self.dropout_rate)
        self.interaction_proj_4 = modules.MLP(self.mid_dim, self.mid_dim, dropout=self.dropout_rate)
        self.aa_res_head = modules.MLP(self.mid_dim, 20, dropout=self.dropout_rate)
        
        # Lambda value mentioned in Figure S6.
        self.register_buffer("alpha_in_sigmoid", torch.tensor(10.0))



    def _load_inputs(self, seq_list, feat_path_list, profile_path_list):
        aa_seq_onehots = aa_sequences_to_padded_onehot(seq_list).to(self.device)

        feat_list = [torch.load(path, weights_only=True)[0] for path in feat_path_list]
        profiles = [np.load(path).T for path in profile_path_list]

        node_reps = pad_feature_matrices(feat_list).to(self.device)
        profiles = pad_feature_matrices(profiles).to(self.device)
        concated_node_reps = torch.cat([node_reps, profiles], dim=-1)

        seq_lengths = torch.tensor([len(seq) for seq in seq_list], device=self.device)
        max_len = int(seq_lengths.max().item())
        node_mask = torch.arange(max_len, device=self.device)[None, :] < seq_lengths[:, None]

        return (
            aa_seq_onehots,
            concated_node_reps,
            seq_lengths,
            node_mask,
            profiles,
        )

    def forward(self, seq_list, feat_path_list, profile_path_list, return_details=False):
        (
            aa_seq_onehots,
            concated_node_reps,
            seq_lengths,
            node_mask,
            profiles,
        ) = self._load_inputs(seq_list, feat_path_list, profile_path_list)

        pe, _ = modules.create_padded_positional_encodings(self.pos_enc, seq_lengths)
        pe = pe.to(self.device)

        B, L_max = concated_node_reps.shape[0], concated_node_reps.shape[1]
        pos = torch.arange(1, L_max + 1, device=self.device).float()   # [L]
        pos = pos.unsqueeze(0).unsqueeze(2).expand(B, -1, 1)    # [B, L, 1]
        pos_feat = torch.log1p(pos)

        refined_node_reps   = self.ProteinMPNN_rep_refine(self.layer_norm_node_rep( concated_node_reps )) + self.MLP_pe_node_rep(pe)
        
        interaction = self.MLP_instead_of_CA(self.layer_norm_node_refined_rep(refined_node_reps))

        x = self.interaction_proj_1(self.layer_norm_interaction_1(interaction) + self.MLP_pe_interact_1(pe))
        x = self.interaction_proj_2(self.layer_norm_interaction_2(x))
        x = self.interaction_proj_3(self.layer_norm_interaction_3(x) + self.MLP_pe_interact_3(pe))
        x = self.interaction_proj_4(self.layer_norm_interaction_4(x))
    
        residue_amino_acid_wise_dG_mat = self.aa_res_head(
            self.layer_norm_residue_amino_acid_wise_dG_mat(x)
        )
        residue_amino_acid_wise_dG_mat = residue_amino_acid_wise_dG_mat * node_mask.unsqueeze(-1).float()

        actual_residue_amino_acid_wise_dG_scores = (
            residue_amino_acid_wise_dG_mat * aa_seq_onehots
        ).sum(dim=-1)
        partial_dG_scores = actual_residue_amino_acid_wise_dG_scores * node_mask.float()

        dG = partial_dG_scores.sum(dim=-1)
        foldability_logit = dG * self.alpha_in_sigmoid

        if not return_details:
            return dG, foldability_logit

        return {
            "dG": dG,
            "foldability_logit": foldability_logit,
            "residue_amino_acid_wise_dG_mat": residue_amino_acid_wise_dG_mat,
        }



class RINAMI_for_dG_regression(nn.Module):
    def __init__(self, device=device, dropout=0.1, ESM_size=320):
        super().__init__()
        self.backbone = BaselineRINAMI(
            device=device,
            dropout=dropout,
            ESM_size=ESM_size,
        )

    def forward(self, seq_list, feat_path_list, profile_path_list, mat_return=False):
        out = self.backbone(
            seq_list,
            feat_path_list,
            profile_path_list,
            return_details=mat_return,
        )
        if not mat_return:
            dG, _ = out
            return dG
        return out["residue_amino_acid_wise_dG_mat"]


class RINAMI_for_foldability_prediction(nn.Module):
    def __init__(self, device=device, dropout=0.1, ESM_size=320):
        super().__init__()
        self.backbone = BaselineRINAMI(
            device=device,
            dropout=dropout,
            ESM_size=ESM_size,
        )

    def forward(self, seq_list, feat_path_list, profile_path_list):
        _, foldability_logit = self.backbone(
            seq_list,
            feat_path_list,
            profile_path_list,
            return_details=False,
        )
        return foldability_logit


