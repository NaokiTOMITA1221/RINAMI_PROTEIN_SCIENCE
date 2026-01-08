
import torch
import numpy as np
from Bio.PDB import PDBParser, DSSP
import os
import glob
import tqdm
import torch
import torch.nn as nn
from protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB
from model_utils import featurize
import subprocess as sb
import sys

def get_protein_mpnn(version='v_48_020.pt'):
    """Loading Pre-trained ProteinMPNN model for structure embeddings"""
    hidden_dim = 128
    num_layers = 3

    model_weight_dir = 'ProteinMPNN_to_get_emb/vanilla_model_weights'
    checkpoint_path = os.path.join(model_weight_dir, version)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim,
                        num_encoder_layers=num_layers, num_decoder_layers=num_layers, k_neighbors=checkpoint['num_edges'], augment_eps=0.0)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if True:
        model.eval()
        # freeze these weights for transfer learning
        for param in model.parameters():
            param.requires_grad = False

    return model

class PDBGraphBuilder:
    def __init__(self, device="cpu"):
        self.device = device
        self.protein_mpnn = get_protein_mpnn()

    def build_graph_from_pdb(self, pdb_path, out_path='../processed_data/temp_ProteinMPNN_node_rep'):
        name = pdb_path.split('/')[-1][:-4]
        pdb = parse_PDB(pdb_path)
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize([pdb[0]], self.device, None, None, None, None, None, None, ca_only=False)
        _, mpnn_embed, _ = self.protein_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)
            

        torch.save(mpnn_embed['h_V'], os.path.join(out_path, f"{name}.pt"))

if __name__ == "__main__":
    args = sys.argv
    pdb_folda = args[-2]
    out_path  = args[-1]
    sb.call(f'mkdir -p {out_path}', shell=True)
    pdbs = glob.glob( os.path.join(pdb_folda, "*.pdb") )
    builder = PDBGraphBuilder(device="cpu")
    for pdb in tqdm.tqdm(pdbs):
        builder.build_graph_from_pdb(pdb, out_path)
