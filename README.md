# RINAMI: Residue-attributed Interpretable Neural network for predicting Absolute folding free energy by Merging structure and sequence Information
!["Figure of architecture"](./Figure/Figure1_RINAMI.png)


## Tested environment:
```
- NVIDIA Driver: 530.41.03
- CUDA: 12.1
- GPU: NVIDIA GeForce RTX 3080 (10GB)
```



## Building an execution environment

Cloning this repository:

    git clone https://github.com/NaokiTOMITA1221/RINAMI_PROTEIN_SCIENCE.git
    cd RINAMI_PROTEIN_SCIENCE


Creation of the environment:

    conda env create -f RINAMI_env.yml 
    conda activate RINAMI_env
    pip install --no-cache-dir \
      torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.4.0+cu121.html


## Using RINAMI
```
cd scripts

# Inference for a single PDB file
python run_inference_for_single_pdb.py [your_pdb_path] 

# If you need the residue-amino-acid-wise ΔG heatmap for a single PDB file
python run_inference_for_single_pdb.py [your_pdb_path] --save-residue-amino-acid-dG-heatmap 


# Inference for multiple PDB files in a directory
python run_inference_for_pdb_batch.py [your_pdb_batch_dir_path] \
    --out-csv [output_csv_path]

# If you need residue-amino-acid-wise ΔG heatmaps for multiple PDB files
python run_inference_for_pdb_batch.py [your_pdb_batch_dir_path] \
    --out-csv [output_csv_path] \
    --save-residue-amino-acid-dG-heatmap \
    --heatmap-out-dir [heatmap_save_dir_path]
```
Note: Please replace `[your_pdb_path]`, `[your_pdb_batch_dir_path]`, `[output_csv_path]`, and `[heatmap_save_dir_path]` with the actual path to a PDB file, the path to a directory containing multiple PDB files, the path to the output CSV file, and the path to the directory where the output heatmaps will be saved, respectively. 
    
## Google Colab implementation of RINAMI is provided on the link below:
https://colab.research.google.com/drive/1N64vgfmstcEQP3i6mS33bS47IH9UVeCs?authuser=1#scrollTo=XNDMAz3ULByd

## Split data, cluster definitions, and benchmark data
Split data, cluster definitions for data splitting, and benchmark data are available from the "processed_data/csv/" directory.

```
processed_data/csv/
                ├── Megascale_wt_clusters_defined_by_mmseqs_25seqid_80coverage.tsv : Cluster definitions for wild-type proteins in the Mega-scale dataset. Clustering was performed using MMseqs2 with 25% sequence identity and 80% coverage.
                |
                ├── split_1/ : Each split directory contains the same five files listed below. 
                |       ├── train_numeric_dG.csv        : Data with numeric dG_ML values used for model training in both the ΔG regression and foldability prediction tasks.
                |       ├── train_dG_gt_5.csv           : Data with dG_ML labels of ">5" used as extreme positive data for model training in the foldability prediction task.
                |       ├── train_dG_lt_minus1.csv      : Data with dG_ML labels of "<-1" used as extreme negative data for model training in the foldability prediction task.
                |       ├── validation_numeric_dG.csv   : Data with numeric dG_ML values used for model validation in both the ΔG regression and foldability prediction tasks.
                |       └── test_numeric_dG.csv         : Data with numeric dG_ML values used for model testing in both the ΔG regression and foldability prediction tasks.
                |
                ├── split_2/     
                |       ├── train_numeric_dG.csv
                |       ├── train_dG_gt_5.csv
                |       ├── train_dG_lt_minus1.csv
                |       ├── validation_numeric_dG.csv
                |       └── test_numeric_dG.csv
                |
                ├── split_3/             
                |       ├── train_numeric_dG.csv
                |       ├── train_dG_gt_5.csv
                |       ├── train_dG_lt_minus1.csv
                |       ├── validation_numeric_dG.csv
                |       └── test_numeric_dG.csv
                |
                ├── Maxwell_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv          : Maxwell benchmark dataset after excluding detected Mega-scale homologs.
                └── Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv : Garcia benchmark dataset after excluding detected Mega-scale homologs.
```

## Preparation for training and testing RINAMI
 
Due to data-size limitations, the structural data used for model training and testing are not included in this repository.

To train and test RINAMI by yourself, the protein structures in the Mega-scale dataset, Maxwell dataset, and Garcia benchmark dataset should be predicted or prepared and saved in the following directories:

    processed_data/Mega_predicted_structure_pdb/ 
    processed_data/Maxwell_predicted_structure_pdb/ 
    processed_data/Garcia_benchmark_predicted_structure_pdb/

In this study, the Mega-scale, Maxwell, and Garcia protein structures were predicted using ESMFold v1 implemented in the esm package version 2.0.0.
Structural prediction for all datasets typically takes about four days with ESMFold.




An example workflow for preparing the structural information used for model training and testing is shown below. Before following this workflow, please confirm that ESMFold is available in your execution environment with a setup equivalent to the one described above.

1. Prepare a FASTA file containing headers that begin with protein names and the corresponding amino acid sequences.
```
Example file: 

processed_data/fasta/Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.fasta

File content: 

>Fd_5S_1_False       # >{protein name}_{foldability label used for the Garcia benchmark test}
LTWEIRVDDEELAEEIERDDPQATVTRKGNTVEVRVTSEDVVKRARERDPEATITRTG
>NF7-02_True
GETTQFDVDENSEKVKRLIRKAGLSEEELKKADIIVIVISRNPEELKRLEEIVRNLGADRIIKLNVDENPEQVRQFAEEAGIPPEKLKRIDYLVVIISKTKEEAKELAERIKRQG
                        .
                        .
                        .
```
Note: When preparing the FASTA file for the Mega-scale dataset, please replace ":" and "|" in each protein name with "_" and remove ".pdb" (For instance, please convert the protein name 'EA|run2_0325_0005.pdb_D1E' into 'EA_run2_0325_0005_D1E').

2. Run the ESMFold prediction.
```
Example for the Garcia dataset: 

cd scripts
python run_ESMFold_prediction.py  \
../processed_data/fasta/Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.fasta \
../processed_data/Garcia_benchmark_predicted_structure_pdb/
```


3. After the structural prediction, please generate ProteinMPNN node representation and ProteinMPNN output profile from the predicted structure and save them into properly made directories, following the process below.
```
python pdb_to_mpnn_node_rep.py ../processed_data/Mega_predicted_structure_pdb ../processed_data/Mega_ProteinMPNN_node_rep
python pdb_to_mpnn_output_profile.py ../processed_data/Mega_predicted_structure_pdb ../processed_data/Mega_ProteinMPNN_output_profile
python pdb_to_mpnn_node_rep.py ../processed_data/Maxwell_predicted_structure_pdb ../processed_data/Maxwell_ProteinMPNN_node_rep
python pdb_to_mpnn_output_profile.py ../processed_data/Maxwell_predicted_structure_pdb ../processed_data/Maxwell_ProteinMPNN_output_profile
python pdb_to_mpnn_node_rep.py ../processed_data/Garcia_benchmark_predicted_structure_pdb ../processed_data/Garcia_benchmark_ProteinMPNN_node_rep
python pdb_to_mpnn_output_profile.py ../processed_data/Garcia_benchmark_predicted_structure_pdb ../processed_data/Garcia_benchmark_ProteinMPNN_output_profile
```
## Training of RINAMI
```
cd RINAMI
bash train_RINAMI.sh 
```
After this execution, the model will be trained and tested once for each of the three splits: split_1, split_2, and split_3.
The parameters of newly trained models will be saved in the directory "pth/pth_RINAMI_trained/".

## Testing RINAMI
```
cd RINAMI

# Benchmark on the Mega-scale test subdataset of the specified data split.
python RINAMI_train_and_test.py [model_parameter_path] Mega_test [split_num]

# Benchmark on the Maxwell dataset using the lowest-performing models.
python RINAMI_train_and_test.py Maxwell_test

# Benchmark on the Garcia dataset using the lowest-performing models.
python RINAMI_train_and_test.py Garcia_test

# Benchmark on the Maxwell dataset using the newly trained model parameters.
python RINAMI_train_and_test.py Maxwell_test USER_TRAINED 

# Benchmark on the Garcia dataset using the newly trained model parameters.
python RINAMI_train_and_test.py Garcia_test USER_TRAINED 
```
Note: Please replace `[model_parameter_path]` and `[split_num]` with the actual model checkpoint path and split number. The `[split_num]` argument in `RINAMI_train_and_test.py` should be set to `1`, `2`, or `3` to use `split_1`, `split_2`, or `split_3`, respectively.
    
## Testing the baseline model

```
cd RINAMI
bash test_Baseline_RINAMI.sh
```
After this execution, the model will be tested once for each of the three splits: split_1, split_2, and split_3.
After these split-wise tests, the Maxwell benchmark will be performed.

## Acknowledgements

## Third-party code and licenses

RINAMI uses or adapts several publicly available resources and software packages.

Protein structures used in this study were predicted using ESMFold implemented in the `esm` package. ProteinMPNN was used to generate structure-based node representations and output profiles from protein structures. Part of the ProteinMPNN-related code and pretrained model parameters are included in `scripts/ProteinMPNN_to_get_emb/` under the ProteinMPNN MIT License, which is provided in `scripts/ProteinMPNN_to_get_emb/LICENSE`.

Some neural-network components in RINAMI, including the MLP module and attention layer, include implementation adapted from the publicly available ChemGLaM repository. ChemGLaM is distributed under the Apache License 2.0, and a copy of the license is provided in `third_party_licenses/ChemGLaM_LICENSE`. The adapted implementation was modified for protein folding free-energy prediction in RINAMI.

This attribution does not imply endorsement, collaboration, or responsibility for RINAMI by the authors of the cited third-party resources.
