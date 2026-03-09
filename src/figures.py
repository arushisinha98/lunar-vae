#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import matplotlib
matplotlib.use('Agg')  # Always use 'Agg' backend for headless environments before importing pyplot
import matplotlib.pyplot as plt

import numpy as np
import torch

def decode(mus_np, model):
    """Decode a numpy array of latent vectors using the VAE decoder."""
    model.eval()
    with torch.no_grad():
        z = torch.from_numpy(mus_np).unsqueeze(-1)  # (N, N_LATENT, 1)
        out = model.decode(z)
    return out.detach().cpu().numpy()

## This needs to be specified - problem dependent
def plot_prediction(inputs_array, outputs_array):
    
    "Plot a network prediction, compare to ground truth and input"
    
    f = plt.figure(figsize=(4*inputs_array.shape[1],4))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for ic in range(inputs_array.shape[1]):
        plt.subplot(1, inputs_array.shape[1], ic+1)
        plt.plot(inputs_array[0, ic, :], color=colors[ic])
        plt.plot(outputs_array[0, ic, :], color=colors[ic])
    
    return f

def plot_result(model, c):
        
    N_LATENT = c["N_LATENT"]
    
    t_grid = np.arange(0,24,0.2)
    
    n_rows = int(np.ceil(N_LATENT/2))
    f = plt.figure(figsize=(np.array([14,5*n_rows])*.7))
    plt.suptitle(c["RUN"],y=0.92)
    for il in range(N_LATENT):
        
        plt.subplot(n_rows,2,il+1)
        
        vals = list(np.linspace(-2.5,2.5,11))
        mus = np.zeros((len(vals),N_LATENT), dtype=np.float32)
        if il==0: vals[5]=0.5 # small instability around 0 latent vector, remove this
        mus[:,0] = 0.5
        for i,s in enumerate(vals): mus[i,il] = s
        
        mu_outputs = decode(mus, model)
        mu_outputs = mu_outputs*c["T_SIGMA"]+c["T_MU"]
        
        for i in range(len(mus)):
            plt.plot(t_grid, mu_outputs[i,0,1:-1], label=mus[i,il])
        
        plt.ylabel("Temperature (K)")
        plt.xlabel("Local lunar time (hours)")
        plt.xticks(np.arange(0,24+6,6),np.arange(0,24+6,6))
        
        if il%2: 
            plt.yticks([])
            plt.ylabel(None)
        if il not in [N_LATENT-1,N_LATENT-2]:
            plt.xticks([])
            plt.xlabel(None)
            
        plt.ylim(70, 400)
        plt.xlim(0,24)

    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    return f