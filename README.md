# RINAMI: Residue-attributed Interpretable Neural network for predicting Absolute folding free energy by Merging structure and sequence Information


Because of the data-size limitation, we put the all data used for model training and testing on the Zenodo strage: https://zenodo.org/records/18171296?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjBlNDU2Y2IzLWYzZGMtNGFhMy04ODk4LTZjYjVhODJkMDhjMCIsImRhdGEiOnt9LCJyYW5kb20iOiJkZGU2ZGMzMDU3ZjJhMGNhNzRjMWNlMjMxOGI3NWMyZCJ9.0PeEYd5-3wrL2Tvy74iufNJivvQt0wXq6XDvkR4Ne17KFRIuuxmLCyZkRRftI5DFbLWXWMRl35-6Rrmay89iGw

After cloning this repository and making a directory named "processed_data" in the cloned repository, please download all data in Zenodo into the created directory following protocols if you need.




After creation of environment, please install pytorch (GPU-build).

 pip install --upgrade --force-reinstall \
   torch==2.4.0+cu121 torchvision torchaudio \
   --index-url https://download.pytorch.org/whl/cu121
