#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:12:01 2018

@author: bmoseley
"""

import torch
from torch.utils.data import Dataset
import numpy as np


# DEFINE DATASET
class TtDataset(Dataset):

    def __init__(self,
                 c,
                 irange=None,
                 verbose=True):
        
        self.c = c # constants data
        self.verbose = verbose
        
        if type(irange) == type(None):
            self.n_examples = c.N_EXAMPLES
            self.irange = np.arange(self.n_examples)
        else:
            self.n_examples = len(irange)
            self.irange = np.array(irange)
        if self.verbose:
            print("%i examples"%(self.n_examples))
            print(self.irange)
        
        # load everything into memory, rather than in __getitem__
        X = np.load(c.DATA_PATH) # shape (N_EXAMPLES, NCHAN, NT)
        print(X.shape, "loaded")
        assert len(X) == self.c.N_EXAMPLES
        assert self.c.T_SHAPE[0] == X.shape[1]
        # make input cyclic
        X = np.concatenate([X[:,:,-1:],X,X[:,:,:1]], axis=2).astype(np.float32)
        print(X.shape, "reshaped")
        self.X = X
        
        self.transform = ToTensor()
        
    def __len__(self): # REQUIRED
        return self.n_examples
    
    def _preprocess(self, profile, i):
        profile = (profile - self.c.T_MU) / self.c.T_SIGMA
        sample = {'inputs': [profile,],
                  'labels': [profile,],
                  'i': i}
        return self.transform(sample)
    
    def __getitem__(self, i):
        return self._preprocess(self.X[self.irange[i]], self.irange[i])

    
class ToTensor(object):
    """Convert numpy arrays in sample to Tensors."""
        
    def __call__(self, sample): # REQUIRED
        transform = {}
        for k in sample:
            if k=="i":
                transform[k] = sample[k]
            elif k in ["inputs","labels"]:
                transform[k] = [torch.from_numpy(array) for array in sample[k]]
            else:
                transform[k] = torch.from_numpy(sample[k])
        return transform


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    from constants import Constants
    from torch.utils.data import DataLoader
    
    c = Constants()
    c.BATCH_SIZE = 4
    print(c)
    
    torch.manual_seed(123)
    
    Dataset = TtDataset
    
    
    traindataset = Dataset(c,
                             irange=np.arange(0,7*c.N_EXAMPLES//10),
                             verbose=True)
    
    testdataset = Dataset(c,
                             irange=np.arange(7*c.N_EXAMPLES//10,c.N_EXAMPLES),
                             verbose=True)
    
    assert len(set(traindataset.irange).intersection(testdataset.irange)) == 0
    
    trainloader = DataLoader(traindataset,
                            batch_size=c.BATCH_SIZE,
                            shuffle=True, # reshuffles data at every epoch
                            num_workers=0, # num_workers = number of multiprocessing workers
                            drop_last=True) # so that each batch is complete
    trainloader_iter = iter(trainloader)
    

    print("TRAIN dataset:")
    for i in range(10):
        sample = traindataset[i]# data sample is read on the fly
        print(i, sample['inputs'][0].size(), sample['labels'][0].size(), sample['i'])
    print(sample['inputs'][0].dtype, sample['labels'][0].dtype)
        
    print("TEST dataset:")
    for i in range(10):
        sample = testdataset[i] # data sample is read on the fly
        print(i, sample['inputs'][0].size(), sample['labels'][0].size(), sample['i'])
    print(sample['inputs'][0].dtype, sample['labels'][0].dtype)
    
    print("BATCHED dataset:")
    for i_batch in range(10):
        sample_batch = next(trainloader_iter)
        print(i_batch, sample_batch['inputs'][0].size(), sample_batch['labels'][0].size(), sample_batch['i'])

    for ib in range(c.BATCH_SIZE):
        x = sample_batch["inputs"][0][ib].detach().cpu().numpy().copy()
        y = sample_batch["labels"][0][ib].detach().cpu().numpy().copy()
        plt.plot(x[0])
        plt.plot(y[0])
    plt.show()