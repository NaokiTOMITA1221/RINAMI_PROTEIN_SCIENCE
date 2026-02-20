# RINAMI: Residue-attributed Interpretable Neural network for predicting Absolute folding free energy by Merging structure and sequence Information
!["Figure of architecture"](./Figure/Figure1_RINAMI.png)
※The illustration of the character RINAMI HIMESAKI is a hand-drawn work created by the first author and is included for non-commercial purposes only.

    Tested environment:
    - NVIDIA Driver: 530.41.03
    - CUDA: 12.1
    - GPU: NVIDIA GeForce RTX 3080 (10GB)





# Building an execution environment

Cloning this repository:

    git clone https://github.com/NaokiTOMITA1221/RINAMI_PROTEIN_SCIENCE.git
    cd RINAMI_PROTEIN_SCIENCE
    mkdir processed_data


Creation of the environment:

    conda env create -f RINAMI_env.yml 
    conda activate RINAMI_env
    pip install --no-cache-dir \
      torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.4.0+cu121.html


# Usage of RINAMI

    cd scripts
    python run_inference.py [your_pdb_path]
    
    
 # Preparation for training and testing RINAMI
 
Because of the data-size limitation, the data used for model training and testing are not put in this repository.
After cloning this repository and making a directory named "processed_data" in the cloned repository, please download "csv.zip" and "decoy_to_seq_dict.json", deposited on Zenodo, into "processed_data" and unzip "csv.zip".

Zenodo strage: https://zenodo.org/records/18221950

When you try the training and test of RINAMI by yourself, structures of proteins in Mega-scale dataset, Maxwell dataset, and Garcia benchmark set should be predicted and saved into:

    processed_data/Mega_predicted_structure_pdb 
    processed_data/Maxwell_predicted_structure_pdb 
    processed_data/Garcia_benchmark_predicted_structure_pdb

respectively. Structural prediction typically takes about one day with ESMFold.

After the structural prediction, please generate ProteinMPNN node representation and ProteinMPNN output profile from the predicted structure and save them into properly made directories, following the process below.
    
    cd scripts
    python pdb_to_mpnn_node_rep.py ../processed_data/Mega_predicted_structure_pdb ../processed_data/Mega_ProteinMPNN_node_rep
    python pdb_to_mpnn_output_profile.py ../processed_data/Mega_predicted_structure_pdb ../processed_data/Mega_ProteinMPNN_output_profile
    python pdb_to_mpnn_node_rep.py ../processed_data/Maxwell_predicted_structure_pdb ../processed_data/Maxwell_ProteinMPNN_node_rep
    python pdb_to_mpnn_output_profile.py ../processed_data/Maxwell_predicted_structure_pdb ../processed_data/Maxwell_ProteinMPNN_output_profile
    python pdb_to_mpnn_node_rep.py ../processed_data/Garcia_benchmark_predicted_structure_pdb ../processed_data/Grcia_benchmark_ProteinMPNN_node_rep
    python pdb_to_mpnn_output_profile.py ../processed_data/Garcia_benchmark_predicted_structure_pdb ../processed_data/Grcia_benchmark_ProteinMPNN_output_profile
    
# Training of RINAMI

    cd model
    bash train_RINAMI.sh 

# Test of RINAMI

    cd model
    python RINAMI_regression_train_and_test.py test_mode [model param path] [test set: "Mega_test", "Maxwell_test", "res_AA_wise_dG_mat"] #Regression task
    python RINAMI_foldability_prediction_train_and_test.py test_mode [model param path] #Foldability prediciton task
    
