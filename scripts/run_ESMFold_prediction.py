import torch
import esm
import tqdm
import biotite.structure.io as bsio
import subprocess as sb
import sys

args = sys.argv

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()


assert len(args)==3, 'Please type: python3 run_ESMFold_prediction.py <input_fasta_path> <output_dir_path>'
    
input_fasta_path = args[1]
output_dir_path  = args[2]


sb.call(f'mkdir -p {output_dir_path}', shell=True)

protein_names = []
seqs = []
for line in open(input_fasta_path):
    if '>' in line:
        protein_names.append(line.strip().replace('>', ''))
    else:
        seqs.append(line.strip())

name_to_seq = {}
seq_to_name = {}
for name, seq in zip(protein_names, seqs):
    name_to_seq[name] = seq
    seq_to_name[seq] = name

for i, name in enumerate(name_to_seq):
    sequence = name_to_seq[name]
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(f"{output_dir_path}/{name}.pdb", "w") as f:
        f.write(output)

    
    struct = bsio.load_structure(f"{output_dir_path}/{name}.pdb", extra_fields=["b_factor"])
    plddt = struct.b_factor.mean()
    sample_ind = i + 1
    all_sample_num = len(name_to_seq)
    print(f'\n{sample_ind}/{all_sample_num}\n{name}: plddt = {plddt}')  
