import json, time, os, sys, glob

if not os.path.isdir("ProteinMPNN"):
  os.system("git clone -q https://github.com/dauparas/ProteinMPNN.git")
sys.path.append('ProteinMPNN')

import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
from protein_mpnn_utils_to_get_emb import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils_to_get_emb import StructureDataset, StructureDatasetPDB, ProteinMPNN

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
#v_48_010=version with 48 edges 0.10A noise
model_name = "v_48_020" #@param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]


backbone_noise=0.00               # Standard deviation of Gaussian noise to add to backbone atoms

path_to_model_weights='ProteinMPNN/vanilla_model_weights'
hidden_dim = 128
num_layers = 3
model_folder_path = path_to_model_weights
if model_folder_path[-1] != '/':
    model_folder_path = model_folder_path + '/'
checkpoint_path = model_folder_path + f'{model_name}.pt'

checkpoint = torch.load(checkpoint_path, map_location=device)
print('Number of edges:', checkpoint['num_edges'])
noise_level_print = checkpoint['noise_level']
print(f'Training noise level: {noise_level_print}A')
model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=backbone_noise, k_neighbors=checkpoint['num_edges'])
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded")

def make_tied_positions_for_homomers(pdb_dict_list):
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1,chain_length+1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i] #needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list
    return my_dict


import re
import numpy as np
import tqdm
import subprocess as sb
#########################
sb.call(f'mkdir ../processed_data/Ikeda_SSP_and_Variants_profile_data', shell=True)
uniprot_IDs = []
for pdb in glob.glob('../processed_data/Ikeda_SSP_and_Variants_pdb/*.pdb'):
    uniprot_IDs.append(pdb.split('/')[-1][:-4])

done_IDs = []
generated_profile = glob.glob('../processed_data/Ikeda_SSP_and_Variants_profile_data/*_profile.npy')
for profile in generated_profile:
  done_IDs.append(profile.split('/')[-1].split('_profile.npy')[0])

uniprot_IDs = list(set(uniprot_IDs) - set(done_IDs))

for ID in tqdm.tqdm(uniprot_IDs):
    try:
      pdb_path = glob.glob(f'../processed_data/Ikeda_SSP_and_Variants_pdb/{ID}.pdb')[0]
      homomer = True #@param {type:"boolean"}
      designed_chain = "A" #@param {type:"string"}
      fixed_chain = "" #@param {type:"string"}

      if designed_chain == "":
        designed_chain_list = []
      else:
        designed_chain_list = re.sub("[^A-Za-z]+",",", designed_chain).split(",")

      if fixed_chain == "":
        fixed_chain_list = []
      else:
        fixed_chain_list = re.sub("[^A-Za-z]+",",", fixed_chain).split(",")

      chain_list = list(set(designed_chain_list + fixed_chain_list))

      #@markdown - specified which chain(s) to design and which chain(s) to keep fixed.
      #@markdown   Use comma:`A,B` to specifiy more than one chain

      #chain = "A" #@param {type:"string"}
      #pdb_path_chains = chain
      ##@markdown - Define which chain to redesign

      #@markdown ### Design Options
      num_seqs = 1 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
      num_seq_per_target = num_seqs

      #@markdown - Sampling temperature for amino acids, T=0.0 means taking argmax, T>>1.0 means sample randomly.
      sampling_temp = "0.1" #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5"]



      save_score=0                      # 0 for False, 1 for True; save score=-log_prob to npy files
      save_probs=0                      # 0 for False, 1 for True; save MPNN predicted probabilites per position
      score_only=0                      # 0 for False, 1 for True; score input backbone-sequence pairs
      conditional_probs_only=0          # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
      conditional_probs_only_backbone=0 # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)

      batch_size=1                      # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
      max_length=20000                  # Max sequence length

      out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
      jsonl_path=''                     # Path to a folder with parsed pdb into jsonl
      omit_AAs='X'                      # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.

      pssm_multi=0.0                    # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
      pssm_threshold=0.0                # A value between -inf + inf to restric per position AAs
      pssm_log_odds_flag=0               # 0 for False, 1 for True
      pssm_bias_flag=0                   # 0 for False, 1 for True


      ##############################################################

      folder_for_outputs = out_folder

      NUM_BATCHES = num_seq_per_target//batch_size
      BATCH_COPIES = batch_size
      temperatures = [float(item) for item in sampling_temp.split()]
      omit_AAs_list = omit_AAs
      alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

      omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

      chain_id_dict = None
      fixed_positions_dict = None
      pssm_dict = None
      omit_AA_dict = None
      bias_AA_dict = None
      tied_positions_dict = None
      bias_by_res_dict = None
      bias_AAs_np = np.zeros(len(alphabet))


      ###############################################################
      pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
      dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=max_length)

      chain_id_dict = {}
      chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)


      if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
      else:
        tied_positions_dict = None


      with torch.no_grad():
        for ix, protein in enumerate(dataset_valid):
          score_list = []
          all_probs_list = []
          all_log_probs_list = []
          S_sample_list = []
          batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
          X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict)
          pssm_log_odds_mask = (pssm_log_odds_all > pssm_threshold).float() #1.0 for true, 0.0 for false
          name_ = batch_clones[0]['name']

          randn_1 = torch.randn(chain_M.shape, device=X.device)
          log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1) #こいつが構造を読んだ後のプロファイルに相当するのでは？
          np.save(f'../processed_data/Ikeda_SSP_and_Variants_profile_data/{ID}_profile.npy', np.exp( log_probs.to('cpu').detach().numpy().copy() )[0, :, :20].T )
    except Exception as e:
      print(f"Error processing {ID}: {e}")
      ##for prob in log_probs[-1]:
      ##    print(prob.shape)
      ##break



