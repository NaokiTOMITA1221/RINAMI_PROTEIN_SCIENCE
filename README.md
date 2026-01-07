# RINAMI: Residue-attributed Interpretable Neural network for predicting Absolute folding free energy by Merging structure and sequence Information
!["Figure of architecture"]("Fig_make_for_paper/dGSS_Figure1.png")

    Tested environment:
    - NVIDIA Driver: 530.41.03
    - CUDA: 12.1
    - GPU: NVIDIA GeForce RTX 3080 (10GB)


Because of the data-size limitation, we put the all data used for model training and testing on the Zenodo strage: https://zenodo.org/records/18171296?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBlNDU2Y2IzLWYzZGMtNGFhMy04ODk4LTZjYjVhODJkMDhjMCIsImRhdGEiOnt9LCJyYW5kb20iOiJkZGU2ZGMzMDU3ZjJhMGNhNzRjMWNlMjMxOGI3NWMyZCJ9.0PeEYd5-3wrL2Tvy74iufNJivvQt0wXq6XDvkR4Ne17KFRIuuxmLCyZkRRftI5DFbLWXWMRl35-6Rrmay89iGw


# Building an execution environment

Cloning this repository:

    git clone https://github.com/NaokiTOMITA1221/RINAMI_PROTEIN_SCIENCE.git
    cd RINAMI_PROTEIN_SCIENCE
    mkdir processed_data

After cloning this repository and making a directory named "processed_data" in the cloned repository, please download all data, deposited on Zenodo, into "processed_data" and unzip if you need.




Creation of the environment:

    conda env create -f RINAMI_env.yml 
    conda activate RINAMI_env
    pip install --no-cache-dir \
      torch-scatter torch-sparse torch-cluster torch-spline-conv \
      -f https://data.pyg.org/whl/torch-2.4.0+cu121.html


# Usage of RINAMI


