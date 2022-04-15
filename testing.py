#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
from utils import getPaths, evaluation_metrics, DynamicChunkSplitData
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import time
import torch
from model import LSTMnet_MeanAtten, LSTMnet_GateAtten, LSTMnet_RnnAtten, LSTMnet_SelfAtten
import argparse



argparse = argparse.ArgumentParser()
argparse.add_argument("-iter", "--iterations", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-atten", "--atten_type", required=True)
args = vars(argparse.parse_args())

# Parameters
iter_max = int(args['iterations'])
batch_size = int(args['batch_size'])
emo_attr = args['emo_attr']
atten_type = args['atten_type']

# Data/Label Dir
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv'
Feat_mean_All = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
Feat_std_All = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']
if emo_attr == 'Act':
    Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Dom':    
    Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Val': 
    Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]    

# Regression Task => Prediction & De-Normalize Target
# MODEL_PATH = './Models/LSTM_iter'+str(iter_max)+'_batch'+str(batch_size)+'_ChunkSeq2One_'+atten_type+'_'+emo_attr+'.pth.tar'
MODEL_PATH = './trained_model_v1.6/LSTM_iter'+str(iter_max)+'_batch'+str(batch_size)+'_ChunkSeq2One_'+atten_type+'_'+emo_attr+'.pth.tar'
if atten_type=='NonAtten':
    model = LSTMnet_MeanAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
elif atten_type=='GatedVec':
    model = LSTMnet_GateAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
elif atten_type=='RnnAttenVec':
    model = LSTMnet_RnnAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
elif atten_type=='SelfAttenVec':
    model = LSTMnet_SelfAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)    
model.load_state_dict(torch.load(MODEL_PATH))
model.cuda()
model.eval()

# Regression Task
test_file_path, test_file_tar = getPaths(label_dir, split_set='Test', emo_attr=emo_attr)

# Testing Data & Label
Test_Pred = []
Test_Label = []
Time_Cost = []
for i in tqdm(range(len(test_file_path))):    
    data = loadmat(root_dir + test_file_path[i].replace('.wav','.mat'))['Audio_data']
    data = data[:,1:] # remove time-info   
    data = (data-Feat_mean_All)/Feat_std_All    # Feature Normalization
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3
    # chunk segmentation pre-processing
    chunk_data = DynamicChunkSplitData([data], m=62, C=11, n=1)
    chunk_data = torch.from_numpy(chunk_data)
    chunk_data = chunk_data.cuda().float() 
    # model flow
    tic = time.time() 
    pred_rsl = model(chunk_data)
    toc = time.time()
    pred_rsl = pred_rsl.data.cpu().numpy()
    # Output prediction results
    Test_Pred.append(pred_rsl)
    Test_Label.append(test_file_tar[i])
    Time_Cost.append((toc-tic)*1000) # unit of time = 10^-3
Test_Pred = np.array(Test_Pred).reshape(-1)
Test_Label = np.array(Test_Label)
Time_Cost = np.array(Time_Cost)

# Regression Task => Prediction & De-Normalize Target
Test_Pred = (Label_std*Test_Pred)+Label_mean

# Output Predict Reulst
pred_Rsl_CCC = evaluation_metrics(Test_Label, Test_Pred)[0]
print('ChunkSeq2One')
print('Iterations: '+str(iter_max))
print('Batch_Size: '+str(batch_size))
print('Model_Type: LSTM')
print('Attention_Type: '+atten_type)
print(emo_attr+'-CCC: '+str(pred_Rsl_CCC))
print('Avg. Time Cost(ms/uttr): '+str(np.mean(Time_Cost)))

