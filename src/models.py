#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 22:00:57 2019

@author: bmoseley
"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, c):
        super().__init__()
        
        self.name = "VAE"
        self.c = c
        
        N_HIDDEN = self.c.N_HIDDEN
        N_LATENT = self.c.N_LATENT
        N_CHANNELS = self.c.T_SHAPE[0]
        
        ## DEFINE WEIGHTS
        
        #Input: (N,Cin,Hin,Win)
        # (N, 1, 128 x 128)
        #Output: (N,Cout,Hout,Wout)
        # (N, 3, 32, 500)

        ## ENCODER
        
        #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        
        self.conv1 = nn.Conv1d(N_CHANNELS, N_HIDDEN, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop1 = nn.Dropout(c.DROPOUT_RATE)
        
        self.conv2 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.conv2_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop2 = nn.Dropout(c.DROPOUT_RATE)
        
        self.conv3 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.conv3_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop3 = nn.Dropout(c.DROPOUT_RATE)
        
        self.conv4 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.conv4_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop4 = nn.Dropout(c.DROPOUT_RATE)

        self.conv5 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.conv5_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop5 = nn.Dropout(c.DROPOUT_RATE)

        self.conv6 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.conv6_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop6 = nn.Dropout(c.DROPOUT_RATE)
        
        self.conv7 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.conv7_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop7 = nn.Dropout(c.DROPOUT_RATE)
        
        self.conv8 = nn.Conv1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.conv8_bn = nn.BatchNorm1d(N_HIDDEN)
        self.drop8 = nn.Dropout(c.DROPOUT_RATE)
        
        self.convMU = nn.Conv1d(N_HIDDEN, N_LATENT, 1, 1, 0)# linear layer
        self.convSD = nn.Conv1d(N_HIDDEN, N_LATENT, 1, 1, 0)# linear layer
        
        ## DECODER

        # (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        
        self.convT1 = nn.ConvTranspose1d(N_LATENT, N_HIDDEN, 2, 2, 0)
        self.convT1_bn = nn.BatchNorm1d(N_HIDDEN)
        self.dropT1 = nn.Dropout(c.DROPOUT_RATE)
        
        self.convT2 = nn.ConvTranspose1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.convT2_bn = nn.BatchNorm1d(N_HIDDEN)
        self.dropT2 = nn.Dropout(c.DROPOUT_RATE)

        self.convT3 = nn.ConvTranspose1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.convT3_bn = nn.BatchNorm1d(N_HIDDEN)
        self.dropT3 = nn.Dropout(c.DROPOUT_RATE)
        
        self.convT4 = nn.ConvTranspose1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.convT4_bn = nn.BatchNorm1d(N_HIDDEN)
        self.dropT4 = nn.Dropout(c.DROPOUT_RATE)
        
        self.convT5 = nn.ConvTranspose1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.convT5_bn = nn.BatchNorm1d(N_HIDDEN)
        self.dropT5 = nn.Dropout(c.DROPOUT_RATE)
        
        self.convT6 = nn.ConvTranspose1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.convT6_bn = nn.BatchNorm1d(N_HIDDEN)
        self.dropT6 = nn.Dropout(c.DROPOUT_RATE)
        
        self.convT7 = nn.ConvTranspose1d(N_HIDDEN, N_HIDDEN, 2, 2, 0)
        self.convT7_bn = nn.BatchNorm1d(N_HIDDEN)
        self.dropT7 = nn.Dropout(c.DROPOUT_RATE)
        
        self.convO = nn.Conv1d(N_HIDDEN, N_CHANNELS, 1, 1, 0)# linear layer

    def encode(self, x):

        ## DEFINE OPERATIONS
        
        x = F.pad(x, (3,3))
        
        #print(x.shape)
        
        x = self.drop1(self.c.ACTIVATION(self.conv1_bn(self.conv1(x))))
        x = self.drop2(self.c.ACTIVATION(self.conv2_bn(self.conv2(x))))
        x = self.drop3(self.c.ACTIVATION(self.conv3_bn(self.conv3(x))))
        x = self.drop4(self.c.ACTIVATION(self.conv4_bn(self.conv4(x))))
        x = self.drop5(self.c.ACTIVATION(self.conv5_bn(self.conv5(x))))
        x = self.drop6(self.c.ACTIVATION(self.conv6_bn(self.conv6(x))))
        x = self.drop7(self.c.ACTIVATION(self.conv7_bn(self.conv7(x))))
        x = self.drop8(self.c.ACTIVATION(self.conv8_bn(self.conv8(x))))

        #print(x.shape)
        
        mu = self.convMU(x)# latent variables
        logvar = self.convSD(x)
        
        return mu,logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    def decode(self, z):

        #print(l.shape)
        
        x = self.dropT1(self.c.ACTIVATION(self.convT1_bn(self.convT1(z))))
        x = self.dropT2(self.c.ACTIVATION(self.convT2_bn(self.convT2(x))))
        x = self.dropT3(self.c.ACTIVATION(self.convT3_bn(self.convT3(x))))
        x = self.dropT4(self.c.ACTIVATION(self.convT4_bn(self.convT4(x))))
        x = self.dropT5(self.c.ACTIVATION(self.convT5_bn(self.convT5(x))))
        x = self.dropT6(self.c.ACTIVATION(self.convT6_bn(self.convT6(x))))
        x = self.dropT7(self.c.ACTIVATION(self.convT7_bn(self.convT7(x))))
        
        #print(x.shape)
        
        x = self.convO(x)# final linear layer
        
        #print(x.shape)
        
        x = x[:,:,3:-3]
        
        #print(x.shape)
        
        return x
        
    def forward(self, x):

        mu,logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    


if __name__ == "__main__":
    
    from constants import Constants
    
    c = Constants()
    
    for Model in [VAE,]:
        model = Model(c)
        
        print("Model: %s"%(model.name))
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters: %i"%(total_params))
        print("Total number of trainable parameters: %i"%(total_trainable_params))
        
        x = torch.zeros((100,)+c.T_SHAPE)
        
        print(x.shape)
        x,m,l = model.forward(x)
        print(x.shape, m.shape, l.shape)
    
    