import torch
import subprocess as sb
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "model"))
from RINAMI_regression_train_and_test import RINAMI
from util import get_sequence_from_single_chain_pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = sys.argv
input_pdb_path = args[-1]
model_param    = '../pth/RINAMI_best_param.pth'

sb.call('mkdir -p ../processed_data/input_pdb', shell=True)
sb.call('mkdir -p ../processed_data/temp_ProteinMPNN_output_profile', shell=True)
sb.call('mkdir -p ../processed_data/temp_ProteinMPNN_node_rep', shell=True)
sb.call(f'cp {input_pdb_path} ../processed_data/input_pdb', shell=True)

model = RINAMI( ESM_size=320 ).to(device)
model.load_state_dict(torch.load(model_param))
model.eval()

sb.call('python ./pdb_to_mpnn_node_rep.py', shell=True)
sb.call('python ./pdb_to_mpnn_output_profile.py', shell=True)

input_data_name = input_pdb_path.split('/')[-1][:-4]
aa_seq                          = get_sequence_from_single_chain_pdb(input_pdb_path)
ProteinMPNN_node_rep_path       = f'../processed_data/temp_ProteinMPNN_node_rep/{input_data_name}.pt'
ProteinMPNN_output_profile_path = f'../processed_data/temp_ProteinMPNN_output_profile/{input_data_name}_profile.npy'

p_dG = model([aa_seq], [ProteinMPNN_node_rep_path], [ProteinMPNN_output_profile_path])[0]

print(f'Input file: {input_pdb_path}')
print(f'Predicted ΔG = {p_dG} [kcal/mol]')

sb.call('rm -r ../processed_data/input_pdb', shell=True)
sb.call('rm -r ../processed_data/temp_ProteinMPNN_output_profile', shell=True)
sb.call('rm -r ../processed_data/temp_ProteinMPNN_node_rep', shell=True)
