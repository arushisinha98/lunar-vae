#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 13:43:14 2018

@author: bmoseley
"""
import socket

import torch

import models
import losses

from datasets import TtDataset
from constantsBase import ConstantsBase



class Constants(ConstantsBase):
    
    def __init__(self, **kwargs):
        "Define default parameters"
        
        ######################################
        ##### GLOBAL CONSTANTS FOR MODEL
        ######################################
        
        self.RUN = "vae_0.20_4_32_f_l2_fin"

        # GPU parameters
        self.DEVICE = 0# cuda device
        
        # Model parameters
        self.MODEL = models.VAE

        self.DROPOUT_RATE = 0.0# probability to drop
        
        self.MODEL_LOAD_PATH = None
        #self.MODEL_LOAD_PATH = "server/models/layers_new_lr1e4_b100_constant8_vvdeep_r_l1/model_03000000.torch"
        
        self.ACTIVATION = torch.relu
        
        #self.N_LATENT = 36
        #self.N_HIDDEN = 64
        self.N_LATENT = 4
        self.N_HIDDEN = 32
        
        # Optimisation parameters
        self.LOSS = losses.vae_loss
        
        self.VAE_BETA = 0.20
        
        self.BATCH_SIZE = 200
        self.LRATE = 1e-3
        self.WEIGHT_DECAY = 0 # L2 weight decay parameter
        self.SCHEDULER_GAMMA = 0.1# multiplicative factor every epoch
        
        # seed
        self.SEED = 123
        
        # training length
        self.N_STEPS = 200000
        
        # CPU parameters
        self.N_CPU_WORKERS = 16# number of multiprocessing workers for DataLoader
        
        self.DATASET = TtDataset
        
        # input dataset properties

        # all data
        #self.DATA_PATH = "data/X_all.npy"
        #self.N_EXAMPLES = 26696
        #self.T_SHAPE = (9, 122)# NCHAN, NT  
        #self.T_MU = np.array([[ 28.54223901  29.29831369 219.43706541 220.96754305 219.36752059
        # 190.59717414 184.93991271 184.15534916 181.20470955]]).T.astype(np.float32)# for L2 -1, 1
        #self.T_SIGMA = np.array([[40.28471199 40.84835589 79.22760694 77.70884204 77.71649932 94.01792034
        # 93.31396689 96.00911998 92.10008878]]).T.astype(np.float32)# NCHAN, 1
        
        # channel 7 filtered
        self.DATA_PATH = "../data_c7_processed/Xf_7_2000.npy"
        self.N_EXAMPLES = 1985693
        self.T_SHAPE = (1, 122)# NCHAN, NT
        self.T_MU = 192.39233778
        self.T_SIGMA = 99.1400795

        ## 3. SUMMARY OUTPUT FREQUENCIES
        self.SUMMARY_FREQ    = 100    # how often to save the summaries, in # steps
        self.TEST_FREQ       = 200    # how often to test the model on test data, in # steps
        self.MODEL_SAVE_FREQ = 50000    # how often to save the model, in # steps

        
        ########
        
        # overwrite with input arguments
        for key in kwargs.keys(): self[key] = kwargs[key]
        
        self.SUMMARY_OUT_DIR = "results/summaries/%s/"%(self.RUN)
        self.MODEL_OUT_DIR = "results/models/%s/"%(self.RUN)
    
        self.HOSTNAME = socket.gethostname().lower()
    



