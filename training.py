#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import torch
import sys
import numpy as np
import os
from utils import cc_coef, DynamicChunkSplitData
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import MspPodcastDataset
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMnet_MeanAtten, LSTMnet_GateAtten, LSTMnet_RnnAtten, LSTMnet_SelfAtten
import argparse  



def collate_fn(batch):  
    data, label = zip(*batch) 
    # LLDs use 16ms hop size to extract features (16ms*62=0.992sec~=1sec)
    chunk_data = DynamicChunkSplitData(data, m=62, C=11, n=1)
    label = np.array(label)
    return torch.from_numpy(chunk_data), torch.from_numpy(label)

def model_validation(model, valid_loader):
    model.eval()
    batch_loss_valid_all = []
    for _, data_batch in enumerate(tqdm(valid_loader, file=sys.stdout)):  
        # Input Tensor Data/Targets
        input_tensor, input_target = data_batch
        input_var = torch.autograd.Variable(input_tensor.cuda()).float()
        input_tar = torch.autograd.Variable(input_target.cuda()).float()         
        # models flow
        pred = model(input_var)
        # loss calculation
        loss = cc_coef(pred, input_tar)
        batch_loss_valid_all.append(loss.data.cpu().numpy())  
        torch.cuda.empty_cache()
    return np.mean(batch_loss_valid_all)
###############################################################################


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
shuffle = True

# LSTM-model loading
if atten_type=='NonAtten':
    model = LSTMnet_MeanAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
elif atten_type=='GatedVec':
    model = LSTMnet_GateAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
elif atten_type=='RnnAttenVec':
    model = LSTMnet_RnnAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
elif atten_type=='SelfAttenVec':
    model = LSTMnet_SelfAtten(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
model.cuda()

# PATH settings
SAVING_PATH = './Models/'
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv'

# creating repo
if not os.path.isdir(SAVING_PATH):
    os.makedirs(SAVING_PATH)

# loading datasets
training_dataset = MspPodcastDataset(root_dir, label_dir, split_set='Train', emo_attr=emo_attr)
validation_dataset = MspPodcastDataset(root_dir, label_dir, split_set='Validation', emo_attr=emo_attr)

# shuffle datasets by generating random indices 
train_indices = list(range(len(training_dataset)))
valid_indices = list(range(len(validation_dataset)))
if shuffle:
    np.random.shuffle(train_indices)

# creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
train_loader = torch.utils.data.DataLoader(training_dataset, 
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=12,
                                           pin_memory=True,
                                           collate_fn=collate_fn)

valid_sampler = SubsetRandomSampler(valid_indices)
valid_loader = torch.utils.data.DataLoader(validation_dataset, 
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=12,
                                           pin_memory=True,
                                           collate_fn=collate_fn)

# create an optimizer for training
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# emotion-recog model training (Iteration-Based)
Iter_trainLoss_All = []
Iter_validLoss_All = []
val_loss_best = 0
iter_count = 0
num_iter_to_valid = 50
while True:
    # stopping criteria
    if iter_count>=iter_max:
        break    
    
    for _, data_batch in enumerate(train_loader):
        # iter setting & record
        model.train()
        iter_count += 1
        # Input Tensor Data/Targets
        input_tensor, input_target = data_batch
        input_var = torch.autograd.Variable(input_tensor.cuda()).float()
        input_tar = torch.autograd.Variable(input_target.cuda()).float()     
        # models flow
        pred = model(input_var)  
        # CCC loss for mean target
        loss = cc_coef(pred, input_tar)
        train_loss = loss.data.cpu().numpy()
        Iter_trainLoss_All.append(train_loss)
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        # clear GPU memory
        torch.cuda.empty_cache()        
        
        # do the model validation every XX iterations
        if iter_count%num_iter_to_valid==0:
            print('validation process')
            val_loss = model_validation(model, valid_loader)
            Iter_validLoss_All.append(val_loss)
            print('Iteration: '+str(iter_count)+' ,Training-loss: '+str(train_loss)+' ,Validation-loss: '+str(val_loss))
            print('=================================================================')
    
            # Checkpoint for saving best model based on val-loss
            if iter_count/num_iter_to_valid==1:
                val_loss_best = val_loss
                torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'LSTM_iter'+str(iter_max)+'_batch'+str(batch_size)+'_ChunkSeq2One_'+atten_type+'_'+emo_attr+'.pth.tar'))
                print("=> Saving the initial best model (Iteration="+str(iter_count)+")")
            else:
                if val_loss_best > val_loss:
                    torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'LSTM_iter'+str(iter_max)+'_batch'+str(batch_size)+'_ChunkSeq2One_'+atten_type+'_'+emo_attr+'.pth.tar'))
                    print("=> Saving a new best model (Iteration="+str(iter_count)+")")
                    print("=> Loss reduction from "+str(val_loss_best)+" to "+str(val_loss) )
                    val_loss_best = val_loss
                else:
                    print("=> Validation Loss did not improve (Iteration="+str(iter_count)+")")
            print('=================================================================')        

# Drawing Loss Curve for Epoch-based and Batch-based
Iter_trainLoss_All = np.mean(np.array(Iter_trainLoss_All[:len(Iter_validLoss_All)*num_iter_to_valid]).reshape(-1, num_iter_to_valid), axis=1).tolist()
plt.title('Epoch-Loss Curve')
plt.plot(Iter_trainLoss_All,color='blue',linewidth=3)
plt.plot(Iter_validLoss_All,color='red',linewidth=3)
plt.savefig(os.path.join(SAVING_PATH, 'LSTM_iter'+str(iter_max)+'_batch'+str(batch_size)+'_ChunkSeq2One_'+atten_type+'_'+emo_attr+'.png'))
#plt.show()

