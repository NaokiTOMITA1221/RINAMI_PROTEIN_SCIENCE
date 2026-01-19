import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import tqdm
import layers
from util import batch_maker_for_inputs, aa_sequences_to_padded_onehot, pad_feature_matrices
import gc
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import torch.nn.functional as F
import json
import glob
import sys
import math
import subprocess as sb
from RINAMI_model_main import RINAMI_for_dG_regression as RINAMI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



        
        

def train_model(model_save_path, trained_model_param=None, num_epochs=5, batch_size=128, dropout=0., pth_ind=None, ESM_size=320):
    ##########################
    # Loading training data  #
    ##########################
    df_train_data = pd.read_csv('../processed_data/csv/mega_train.csv')
    struct_list_train_data   = []
    mpnn_profile_train_data  = []

    for name, wt_name in zip(df_train_data['name'], df_train_data['WT_name']):
        mutant_label = name.split('.pdb')[0] + name.split('.pdb')[1]
        label        = wt_name.split('.pdb')[0]
        struct_list_train_data.append(f'../processed_data/Mega_ProteinMPNN_node_rep/{name}.pt')
        mpnn_profile_train_data.append(f'../processed_data/Mega_ProteinMPNN_output_profile/{name}_profile.npy')
       
    seq_list_train_data = list(df_train_data['aa_seq'])
    dG_list_train_data = list(df_train_data['dG_ML'])
    
    ############################
    # Loading validation data  #
    ############################
    df_val_data            = pd.read_csv('../processed_data/csv/mega_val.csv')
    struct_list_val_data   = []
    mpnn_profile_val_data  = []

    for name, wt_name in zip(df_val_data['name'], df_val_data['WT_name']):
        mutant_label = name.split('.pdb')[0] + name.split('.pdb')[1]
        label        = wt_name.split('.pdb')[0]
        struct_list_val_data.append(f'../processed_data/Mega_ProteinMPNN_node_rep/{name}.pt')
        mpnn_profile_val_data.append(f'../processed_data/Mega_ProteinMPNN_output_profile/{name}_profile.npy')

    seq_list_val_data = list(df_val_data['aa_seq'])
    dG_list_val_data = list(df_val_data['dG_ML'])


    ###################
    # Loading RINAMI  #
    ###################
    if trained_model_param==None:
        model = RINAMI(dropout=dropout, ESM_size=ESM_size).to(device)
    else:
        model = RINAMI(dropout=dropout, ESM_size=ESM_size).to(device)
        model.load_state_dict(torch.load(trained_model_param))
    
    ####################################
    # Loss function for dG regression  #
    ####################################
    criterion_1 = nn.HuberLoss()
    
    
    #########################
    # Setting for training  #
    #########################
    head_params   = []
    encoder_params = []
    for n,p in model.named_parameters():
        if any(k in n for k in ["interaction_converter", "MLP_pe_", "ESM_rep_projection", "ProteinMPNN_profile_projection"]):
            head_params.append(p)
        else:
            encoder_params.append(p)
    
    optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": 1e-5, "weight_decay": 0.01},
                {"params": head_params,    "lr": 5e-4, "weight_decay": 0.01},
              ])
    
    total_steps = (len(seq_list_train_data) + batch_size - 1)//batch_size * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps),
                                            num_training_steps=total_steps)
    

    #################
    # Training loop #
    #################
    for epoch in range(num_epochs):
        if epoch > 2:
            continue
        model.train()
        training_loss = 0.0
        batch_list_train = batch_maker_for_inputs(seq_list_train_data, struct_list_train_data, mpnn_profile_train_data, dG_list_train_data, batch_size)
        #Training steps
        for batch in tqdm.tqdm(batch_list_train):
            aa_seq_batch   = batch[0]
            struct_batch   = batch[1]
            profile_batch  = batch[2]
            dG_batch       = torch.tensor(batch[3], dtype=torch.float32).to(device)
        
            output   = model(aa_seq_batch, struct_batch, profile_batch)

            loss = criterion_1(output, dG_batch)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            training_loss += loss.item()
            
            del output, dG_batch, loss
            torch.cuda.empty_cache() 
        
        loss_avg = training_loss / int(len(seq_list_train_data)/batch_size)
        
        
        
        #Validation step
        p_dG_list = []
        e_dG_list = []
        
        model.eval()
        validation_loss = 0.0
        batch_list_val = batch_maker_for_inputs(seq_list_val_data, struct_list_val_data, mpnn_profile_val_data, dG_list_val_data, batch_size)
        

        with torch.no_grad():
            for batch in tqdm.tqdm(batch_list_val):
                aa_seq_batch   = batch[0]
                struct_batch   = batch[1]
                profile_batch  = batch[2]
                dG_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)

                output = model(aa_seq_batch, struct_batch, profile_batch)

                loss = criterion_1(output, dG_batch)
                validation_loss += loss.item()
            
                
                for p_dG, e_dG in zip(output.to('cpu'), dG_batch.to('cpu')):
                    p_dG_list.append(p_dG)
                    e_dG_list.append(e_dG)


        loss_avg_test = validation_loss / int(len(seq_list_val_data)/batch_size)
        corr          = np.corrcoef(p_dG_list,e_dG_list)[0,1]
        
        ##############
        # Monitoring #
        ##############
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss_avg:.10f}, Validation Loss: {loss_avg_test:.10f}, Correlation of validation: {corr}")
        
        ####################
        # Save checkpoints #
        ####################
        temp_model_save_path = model_save_path.replace('.pth',f'_{epoch}epoch.pth')
        torch.save(model.state_dict(), temp_model_save_path)

    torch.save(model.state_dict(), model_save_path)



def test_model(trained_model_param, ESM_size, batch_size=16):
    ######################
    # Loading test data  #
    ######################
    df_test_data            = pd.read_csv('../processed_data/csv/mega_test.csv')
    struct_list_test_data   = []
    mpnn_profile_test_data  = []
    rosetta_score_test_data = []

    for name, wt_name in zip(df_test_data['name'], df_test_data['WT_name']):
        mutant_label = name.split('.pdb')[0] + name.split('.pdb')[1]
        label        = wt_name.split('.pdb')[0]
        struct_list_test_data.append(f'../processed_data/Mega_ProteinMPNN_node_rep/{name}.pt')
        mpnn_profile_test_data.append(f'../processed_data/Mega_ProteinMPNN_output_profile/{name}_profile.npy')

    seq_list_test_data = list(df_test_data['aa_seq'])
    dG_list_test_data = list(df_test_data['dG_ML'])


    model = RINAMI( ESM_size=ESM_size).to(device)
    model.load_state_dict(torch.load(trained_model_param))


    #Test step
    p_dG_list = []
    e_dG_list = []

    model.eval()
    batch_list_test = batch_maker_for_inputs(seq_list_test_data, struct_list_test_data, mpnn_profile_test_data, dG_list_test_data, batch_size)


    with torch.no_grad():
        for batch in tqdm.tqdm(batch_list_test):
            aa_seq_batch = batch[0]
            struct_batch = batch[1]
            profile_batch  = batch[2]
            dG_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)
            output = model(aa_seq_batch, struct_batch, profile_batch)



            for p_dG, e_dG in zip(output.to('cpu'), dG_batch.to('cpu')):
                p_dG_list.append(p_dG)
                e_dG_list.append(e_dG)


        corr          = np.corrcoef(p_dG_list,e_dG_list)[0,1]

        print(f"Correlation of test: {corr}")



def pdb_id_to_dGmat(trained_model_param, ESM_size):    
    sb.call(f'mkdir -p ../processed_data/wt_dG_mat', shell=True)
    
    df                 = pd.read_csv('../processed_data/csv/mega_test.csv')
    struct_list_data   = []
    mpnn_profile_data  = []
    seq_list_data      = []
    wt_names           = []

    for name, wt_name, mut_type, aa_seq, dG in zip(df['name'], df['WT_name'], df['mut_type'], df['aa_seq'], df['dG_ML']):
        if mut_type == 'wt': 
            mutant_label = name.split('.pdb')[0] + name.split('.pdb')[1]
            label = wt_name.split('.pdb')[0]
            struct_list_data.append(f'../processed_data/Mega_ProteinMPNN_node_rep/{name}.pt')
            mpnn_profile_data.append(f'../processed_data/Mega_ProteinMPNN_output_profile/{name}_profile.npy')
            seq_list_data.append(aa_seq)
            wt_names.append(name)

    model = RINAMI( ESM_size=ESM_size).to(device)
    model.load_state_dict(torch.load(trained_model_param))
    
    batch_list = batch_maker_for_inputs(seq_list_data, struct_list_data, mpnn_profile_data, wt_names, batch_size=1)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(batch_list):
            aa_seq_batch   = batch[0]
            struct_batch   = batch[1]
            profile_batch  = batch[2]
            wt_names_batch = batch[3]
            dG_mat         = model(aa_seq_batch, struct_batch, profile_batch, mat_return=True).to('cpu')[0].numpy()
            wt_name        = wt_names_batch[0]
            np.save(f'../processed_data/wt_dG_mat/{wt_name}_res_wise_dG.npy', dG_mat)
        

def test_model_with_Maxwell(trained_model_param, ESM_size, num_epochs=1, batch_size=1, lr=3e-5, dropout=0., pth_ind=None, def_dim=12, heads=8):
    ######################
    # Loading test data  #
    ######################
    df                 = pd.read_csv('../processed_data/csv/maxwell2007_sequences.csv')
    struct_list_data   = []
    mpnn_profile_data  = []
    
    for protein_id in df['id']:
        struct_list_data.append(glob.glob(f'../processed_data/Maxwell_ProteinMPNN_node_rep/{protein_id}*.pt')[0])
        mpnn_profile_data.append(glob.glob(f'../processed_data/Maxwell_ProteinMPNN_output_profile/{protein_id}*.npy')[0])

    seq_list_data = list(df['sequence'])
    dG_list_data = [dG*0.239006 for dG in df['dg']] #Convert: [J/mol] -> [kcal/mol]
    
    model = RINAMI( ESM_size=ESM_size).to(device)
    model.load_state_dict(torch.load(trained_model_param))
    
    print('p_dG, e_dG')
    for epoch in range(num_epochs):

        #Test step
        p_dG_list = []
        e_dG_list = []

        model.eval()
        validation_loss = 0.0
        batch_list_val = batch_maker_for_inputs(seq_list_data, struct_list_data, mpnn_profile_data, dG_list_data, batch_size)


        with torch.no_grad():
            for batch in batch_list_val:
                aa_seq_batch  = batch[0]
                struct_batch  = batch[1]
                profile_batch = batch[2]
                dG_batch = torch.tensor(batch[3], dtype=torch.float32).to(device)

                output = model(aa_seq_batch, struct_batch, profile_batch)

                for p_dG, e_dG in zip(output.to('cpu'), dG_batch.to('cpu')):
                    print(round(float(p_dG), 2), round(float(e_dG), 2))
                    p_dG_list.append(p_dG)
                    e_dG_list.append(e_dG)
    
        corr          = np.corrcoef(p_dG_list,e_dG_list)[0,1]
        print(f"\nCorrelation of Maxwell test: {corr}")




                   



if __name__ == "__main__":
   """
    Testing  : python3 RINAMI_regression_train_and_test.py test_mode <model param path> <test set: "Mega_test", "Maxwell_test", "res_AA_wise_dG_mat">
   """
   args = sys.argv
   
       
   if len(args) == 3:
       ESM_dim = 320
       print(f'training step')
       train_model(args[1], trained_model_param=args[2], num_epochs=3, dropout=0., ESM_size=ESM_dim)
    
        
   
   elif len(args) == 4:
        ESM_dim = 320
        trained_model_path = args[-2]
        test_mode          = args[-1]
        
        if test_mode == 'Mega_test':
            print('Test mode: Mega_test')
            test_model(trained_model_path, ESM_size=ESM_dim)
        elif test_mode == 'Maxwell_test':
            print('Test mode: Maxwell_test')
            test_model_with_Maxwell(trained_model_path, ESM_size=ESM_dim)
        elif test_mode == 'res_AA_wise_dG_mat':
            print('Extracting residue-amino-acid-wise dG matrics')
            pdb_id_to_dGmat(trained_model_path, ESM_size=ESM_dim)

