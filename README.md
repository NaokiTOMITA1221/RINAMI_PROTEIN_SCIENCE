# RINAMI: Residue-attributed Interpretable Neural network for predicting Absolute folding free energy by Merging structure and sequence Information


Because of the data-size limitation, we put the all data used for model training and testing on the Zenodo strage whose link was descrived below: (URL of Zenodo)

After cloning this repository and making a directory named "processed_data" in the cloned repository, please download all data in Zenodo into the created directory following protocols.




After creation of environment, please install pytorch (GPU-build).

 pip install --upgrade --force-reinstall \
   torch==2.4.0+cu121 torchvision torchaudio \
   --index-url https://download.pytorch.org/whl/cu121
