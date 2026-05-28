# RINAMI: Residue-attributed Interpretable Neural network for predicting Absolute folding free energy by Merging structure and sequence Information
!["Figure of architecture"](./Figure/Figure1_RINAMI.png)


    Tested environment:
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

    cd scripts
    python run_inference.py [your_pdb_path] 
    
    python run_inference.py [your_pdb_path] --save-residue-amino-acid-dG-heatmap #If you need the residue-amino-acid-wise ΔG matrix.
    
    
# Google Colab implementation of RINAMI is provided on the link below:
https://colab.research.google.com/drive/1N64vgfmstcEQP3i6mS33bS47IH9UVeCs?authuser=1#scrollTo=XNDMAz3ULByd

# Split data, cluster definition, and benchmark data 
Split data, cluster definitions for data splitting, and benchmark data are available from the "processed_data/csv/" directory.

processed_data/csv/

　・Megascale_wt_clusters_defined_by_mmseqs_25seqid_80coverage.tsv
　
　　　Cluster definitions for wild-type proteins in the Mega-scale dataset.
　　　Clustering was performed using MMseqs2 with 25% sequence identity and 80% coverage.

  ・split_1/, split_2/, split_3/
      
    Train/validation/test split files used in this study.
      
    ・train_numeric_dG.csv
       
       Data with numeric dG_ML values used for model training in both the ΔG regression and foldability prediction tasks.
    ・train_dG_gt_5.csv
       
       Data with dG_ML labels of ">5" used as extreme positive data for model training in the foldability prediction task.
    ・train_dG_lt_minus1.csv
       
       Data with dG_ML labels of "-1>" used as extreme negative data for model training in the foldability prediction task.
    ・validation_numeric_dG.csv
       
       Data with numeric dG_ML values used for model validation in both the ΔG regression and foldability prediction tasks.
    ・test_numeric_dG.csv
       
       Data with numeric dG_ML values used for model testing in both the ΔG regression and foldability prediction tasks.
         
    are put in each split folder.

  ・Maxwell_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv
      
    Maxwell benchmark dataset after excluding detected Mega-scale homologs.

  ・Garcia_zero_shot_without_detected_megascale_homologs_mmseqs_25seqid_80coverage.csv
      
    Garcia benchmark dataset after excluding detected Mega-scale homologs.
      
# Preparation for training and testing RINAMI
 
Because of the data-size limitation, the structural data used for model training and testing are not put in this repository.

When you try the training and test of RINAMI by yourself, structures of proteins in Mega-scale dataset, Maxwell dataset, and Garcia benchmark set should be predicted and saved into:

    processed_data/Mega_predicted_structure_pdb 
    processed_data/Maxwell_predicted_structure_pdb 
    processed_data/Garcia_benchmark_predicted_structure_pdb

respectively. Structural prediction typically takes about four day with ESMFold.

※ When predicting the structures of Mega-scale examples, please replace ":" and "|" in each example name with "_". (For instance, please save the predicted structure of the Mega-scale example, 'EA|run2_0325_0005.pdb_D1E', as 'EA_run2_0325_0005_D1E.pdb')

After the structural prediction, please generate ProteinMPNN node representation and ProteinMPNN output profile from the predicted structure and save them into properly made directories, following the process below.
    
    cd scripts
    python pdb_to_mpnn_node_rep.py ../processed_data/Mega_predicted_structure_pdb ../processed_data/Mega_ProteinMPNN_node_rep
    python pdb_to_mpnn_output_profile.py ../processed_data/Mega_predicted_structure_pdb ../processed_data/Mega_ProteinMPNN_output_profile
    python pdb_to_mpnn_node_rep.py ../processed_data/Maxwell_predicted_structure_pdb ../processed_data/Maxwell_ProteinMPNN_node_rep
    python pdb_to_mpnn_output_profile.py ../processed_data/Maxwell_predicted_structure_pdb ../processed_data/Maxwell_ProteinMPNN_output_profile
    python pdb_to_mpnn_node_rep.py ../processed_data/Garcia_benchmark_predicted_structure_pdb ../processed_data/Garcia_benchmark_ProteinMPNN_node_rep
    python pdb_to_mpnn_output_profile.py ../processed_data/Garcia_benchmark_predicted_structure_pdb ../processed_data/Garcia_benchmark_ProteinMPNN_output_profile
    
# Training of RINAMI

    cd RINAMI
    bash train_RINAMI.sh 
 The parameters of newly trained models will be saved in the directory "pth/pth_RINAMI_trained/".

# Test of RINAMI

    cd RINAMI
    python3 RINAMI_train_and_test.py [model param path] Mega_test [split num] [dummy_outdir path] #Mega-scale test
    python3 RINAMI_train_and_test.py [model param path] export_interpretability [split num] [out_dir path] #Extracting residue-amino-acid-wise ΔG matrix of wild-type proteins in Mega-scale validation and test subdatasets
    python3 RINAMI_train_and_test.py Maxwell_test USER_TRAINED #Maxwell benchmark test
    python3 RINAMI_train_and_test.py Garcia_test USER_TRAINED #Garcia benchmark test
    
