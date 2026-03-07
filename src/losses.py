#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:57:42 2018

@author: bmoseley
"""

import torch

# DEFINES VARIOUS LOSS FUNCTIONS

# Trainer passes arguments in the following order: loss(*labels, *outputs, constants)

def l2_mean_loss(a,b,*args):
    return torch.mean((a-b)**2)

def l2_sum_loss(a,b,*args):
    return torch.sum((a-b)**2)

def l1_mean_loss(a,b,*args):
    return torch.mean(torch.abs(a - b))

def l1_sum_loss(a,b,*args):
    return torch.sum(torch.abs(a - b))

def kld_mean_loss(x_true, x, mu, logvar, c):
    # for QC
    
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return KLD

def vae_loss(x_true, x, mu, logvar, c):
    
    # notes - output needs to be between 0,1 for binary_cross_entropy
    # binary cross entropy for bernouli
    # mse for gaussian
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    #BCE = F.binary_cross_entropy(x, x_true, reduction='sum')
    BCE = l2_sum_loss(x, x_true)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + c.VAE_BETA*KLD


if __name__ == "__main__":
    
    a = torch.ones((10,20))
    b = torch.ones((10,20)).mul_(0.1) # (in place)
    
    print(l2_sum_loss(a,b))
    
    print(l2_mean_loss(a,b))
    print(l1_mean_loss(a,b))