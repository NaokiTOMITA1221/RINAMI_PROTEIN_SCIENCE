# RINAMI: Residue-attributed Interpretable Neural network for predicting Absolute folding free energy by Merging structure and sequence Information
!["Figure of architecture"](./Figure/Figure1_RINAMI.png)


    Tested environment:
    - NVIDIA Driver: 545.23.06
    - CUDA: 12.3
    - GPU: NVIDIA GeForce RTX 3080 Ti (12GB)
            
            and
            
    - NVIDIA Driver: 530.41.03
    - CUDA: 12.1
    - GPU: NVIDIA GeForce RTX 3080 (10GB)




# Building an execution environment

Cloning this repository:

    git clone https://github.com/NaokiTOMITA1221/RINAMI_PROTEIN_SCIENCE.git
    cd RINAMI_PROTEIN_SCIENCE


Creation of the environment:

    conda env create -f RINAMI_env.yml 
    conda activate RINAMI_env
    pip install --no-cache-dir \
      torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.4.0+cu121.html


# Usage of RINAMI
```
cd scripts
python run_inference.py [your_pdb_path] 
    
# If you need the residue-amino-acid-wise ΔG matrix.
python run_inference.py [your_pdb_path] --save-residue-amino-acid-dG-heatmap 
```
    
# Google Colab implementation of RINAMI is provided on the link below:
https://colab.research.google.com/drive/1N64vgfmstcEQP3i6mS33bS47IH9UVeCs?authuser=1#scrollTo=XNDMAz3ULByd

# Split data, cluster definition, and benchmark data 
Split data, cluster definitions for data splitting, and benchmark data are available from the "processed_data/csv/" directory.

```
processed_data/csv/
                ├── Megascale_wt_clusters_defined_by_mmseqs_25seqid_80coverage.tsv : Cluster definitions for wild-type proteins in the Mega-scale dataset. Clustering was performed using MMseqs2 with 25% sequence identity and 80% coverage.
                |
                ├── split_1/             
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

# Preparation for training and testing RINAMI
 
Due to data-size limitations, the structural data used for model training and testing are not included in this repository.

To train and test RINAMI by yourself, the protein structures in the Mega-scale dataset, Maxwell dataset, and Garcia benchmark dataset should be predicted or prepared and saved in the following directories:

    processed_data/Mega_predicted_structure_pdb/ 
    processed_data/Maxwell_predicted_structure_pdb/ 
    processed_data/Garcia_benchmark_predicted_structure_pdb/

In this study, the Mega-scale and Garcia protein structures were predicted using ESMFold v1 implemented in the esm package version 2.0.0. The Maxwell protein structures were obtained from the "paper_SI/maxwell2009/af2_best/" directory in Cagiada et al.’s GitHub repository: "https://github.com/KULL-Centre/_2024_cagiada_stability". Structural prediction for all datasets typically takes about four days with ESMFold.




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
※ When preparing the FASTA file for the Mega-scale dataset, please replace ":" and "|" in each protein name with "_" and remove ".pdb" (For instance, please convert the protein name 'EA|run2_0325_0005.pdb_D1E' into 'EA_run2_0325_0005_D1E').

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
# Training of RINAMI
```
cd RINAMI
bash train_RINAMI.sh 
```
After this execution, the model will be trained and tested once for each of the three splits: split_1, split_2, and split_3.
The parameters of newly trained models will be saved in the directory "pth/pth_RINAMI_trained/".

# Test of RINAMI
```
cd RINAMI

# Benchmark on the Mega-scale test subdataset of the specified data split.
python RINAMI_train_and_test.py [model param path] Mega_test [split num] 

# Benchmark on the Maxwell dataset using the lowest-performing models.
python RINAMI_train_and_test.py Maxwell_test

# Benchmark on the Garcia dataset using the lowest-performing models.
python RINAMI_train_and_test.py Garcia_test

# Benchmark on the Maxwell dataset using the newly trained model parameters.
python RINAMI_train_and_test.py Maxwell_test USER_TRAINED 

# Benchmark on the Garcia dataset using the newly trained model parameters.
python RINAMI_train_and_test.py Garcia_test USER_TRAINED 
```
※ The argument [split num] in RINAMI_train_and_test.py should be set to 1, 2, or 3 to use split_1, split_2, or split_3, respectively.
    
# Test of Baseline model

```
cd RINAMI
bash test_Baseline_RINAMI.sh
```
After this execution, the model will be tested once for each of the three splits: split_1, split_2, and split_3.
The Maxwell benchmark then will start.
