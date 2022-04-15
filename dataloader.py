#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: winston lin
"""
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import getPaths


class MspPodcastDataset(Dataset):
    """MSP-Podcast Emotion dataset."""

    def __init__(self, root_dir, label_dir, split_set, emo_attr):
        # Parameters
        self.root_dir = root_dir
        # Loading Label Distribution
        self._paths, self._labels = getPaths(label_dir, split_set=split_set, emo_attr=emo_attr)       
        # Loading Norm-Feature
        self.Feat_mean_All = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std_All = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        # Loading Norm-Label
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0] 
        elif emo_attr == 'Dom':
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]  
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0] 

    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading Data & Normalization
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data']
        data = data[:,1:] # remove time-info
        # z-norm and bounded in -3~3 range (i.e., 99.5% values coverage)       
        data = (data-self.Feat_mean_All)/self.Feat_std_All
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # Label Normalization
        label = (self._labels[idx]-self.Label_mean)/self.Label_std
        return data, label

