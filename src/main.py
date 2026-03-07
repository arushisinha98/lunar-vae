#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:04:43 2018

@author: bmoseley
"""


import sys
import os
import time

import matplotlib
if 'linux' in sys.platform.lower(): matplotlib.use('Agg')# use a non-interactive backend (ie plotting without windows)
import matplotlib.pyplot as plt

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from constants import Constants
import losses
from torch_utils import get_weights, get_weights_update_percent

def decode(mus_np, model):
    """Decode a numpy array of latent vectors using the VAE decoder."""
    model.eval()
    with torch.no_grad():
        z = torch.from_numpy(mus_np).unsqueeze(-1)  # (N, N_LATENT, 1)
        out = model.decode(z)
    return out.detach().cpu().numpy()

## This needs to be specified - problem dependent
def plot_result(inputs_array, outputs_array, labels_array, model, c):
    
    "Plot a network prediction, compare to ground truth and input"
    
    '''
    f = plt.figure(figsize=(4*inputs_array.shape[1],4))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for ic in range(inputs_array.shape[1]):
        plt.subplot(1, inputs_array.shape[1], ic+1)
        plt.plot(inputs_array[0, ic, :], color=colors[ic])
        plt.plot(outputs_array[0, ic, :], color=colors[ic])
    ''' 
        
    N_LATENT = c["N_LATENT"]
    
    t_grid = np.arange(0,24,0.2)
    
    
    n_rows = int(np.ceil(N_LATENT/2))
    f = plt.figure(figsize=(np.array([14,5*n_rows])*.7))
    plt.suptitle(c["RUN"],y=0.92)
    for il in range(N_LATENT):
        
        plt.subplot(n_rows,2,il+1)
        
        vals = list(np.linspace(-2.5,2.5,11))
        mus = np.zeros((len(vals),N_LATENT), dtype=np.float32)
        if il==0: vals[5]=0.5# small instability around 0 latent vector, remove this
        mus[:,0] = 0.5
        for i,s in enumerate(vals): mus[i,il] = s
        
        mu_outputs = decode(mus, model)
        mu_outputs = mu_outputs*c["T_SIGMA"]+c["T_MU"]
        
        for i in range(len(mus)):
            plt.plot(t_grid, mu_outputs[i,0,1:-1], label=mus[i,il])
        #plt.title(il)
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
        


class Trainer:
    "Generic model trainer class"
    
    def __init__(self, c):
        "Initialise torch, output directories, training dataset and model"
        
        
        ## INITIALISE
        
        # set seed
        if c.SEED == None: c.SEED = torch.initial_seed()
        else: torch.manual_seed(c.SEED)# likely independent of numpy
        np.random.seed(c.SEED)
                       
        # clear directories
        c.get_outdirs()
        c.save_constants_file()# saves torch seed too
        print(c)
        
        # set device/ threads
        device = torch.device("cuda:%i"%(c.DEVICE) if torch.cuda.is_available() else "cpu")
        print("Device: %s"%(device))
        torch.backends.cudnn.benchmark = False#let cudnn find the best algorithm to use for your hardware (not good for dynamic nets)
        torch.set_num_threads(1)# for main inference
        
        print("Main thread ID: %i"%os.getpid())
        print("Number of CPU threads: ", torch.get_num_threads())
        print("Torch seed: ", torch.initial_seed())
        
        # initialise summary writer
        writer = SummaryWriter(c.SUMMARY_OUT_DIR)
        writer.add_text("constants", str(c).replace("\n","  \n"))# uses markdown

        ### DEFINE TRAIN/TEST DATASETS
        
        # split dataset 80:20
        irange = np.arange(0, c.N_EXAMPLES)
        np.random.shuffle(irange)# randomly shuffle the indicies (in place) before splitting. To get diversity in train/test split.
        traindataset = c.DATASET(c,
                                 irange=irange[0:(8*c.N_EXAMPLES//10)],
                                 verbose=True)
        testdataset = c.DATASET(c,
                                irange=irange[(8*c.N_EXAMPLES//10):c.N_EXAMPLES],
                                verbose=True)
        assert len(set(traindataset.irange).intersection(testdataset.irange)) == 0# make sure examples aren't shared!
        
        #### DEFINE MODEL
        
        model = c.MODEL(c)
        
        # load previous weights
        if c.MODEL_LOAD_PATH != None:
            cp = torch.load(c.MODEL_LOAD_PATH,
                            map_location=torch.device('cpu'))# remap tensors from gpu to cpu if needed
            model.load_state_dict(cp['model_state_dict'])
            ioffset = cp["i"]
            print("Loaded model weights from: %s"%(c.MODEL_LOAD_PATH))
        else: ioffset = 0
        
        # print out parameters
        #writer.add_graph(model, torch.zeros((1,)+c.VELOCITY_SHAPE))# write graph before placing on GPU
        print()
        print("Model: %s"%(model.name))
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters: %i"%(total_params))
        print("Total number of trainable parameters: %i"%(total_trainable_params))
        #for p in model.parameters(): print(p.size(), p.numel())
        
        model.to(device)

        self.c, self.device, self.writer = c, device, writer
        self.traindataset, self.testdataset = traindataset, testdataset
        self.model, self.ioffset = model, ioffset

    def train(self):
        "train model"
        
        c, device, writer = self.c, self.device, self.writer
        traindataset, testdataset = self.traindataset, self.testdataset
        model, ioffset = self.model, self.ioffset
        
        ### TRAIN
        
        print()
        print("Training..")
        
        N_BATCHES = len(traindataset)//c.BATCH_SIZE
        N_EPOCHS = int(np.ceil(c.N_STEPS/N_BATCHES))
        
        trainloader = DataLoader(traindataset,
                        batch_size=c.BATCH_SIZE,#shuffle=True, # reshuffles data at every epoch (RandomSampler)
                        sampler=RandomSampler(traindataset, replacement=True),# randomly sample with replacement
                        num_workers=0,# num_workers = spawns multiprocessing subprocess workers
                        drop_last=True,# so that each batch is complete
                        timeout=300)# timeout after 5 mins of no data loading
        
        testloader = DataLoader(testdataset,
                        batch_size=c.BATCH_SIZE,#shuffle=True, # reshuffles data at every epoch (RandomSampler)
                        sampler=RandomSampler(testdataset, replacement=True),# randomly sample with replacement
                        num_workers=0,# num_workers = spawns multiprocessing subprocess workers
                        drop_last=True,# so that each batch is complete
                        timeout=300)# timeout after 5 mins of no data loading
        
        testloader_iterator = iter(testloader)
        trainloader_iterator = iter(trainloader)
        assert len(trainloader_iterator) == N_BATCHES
        
        #optimizer = torch.optim.SGD(model.parameters(), lr=c.LRATE, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE, weight_decay=c.WEIGHT_DECAY)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(np.log(c.SCHEDULER_GAMMA)/(100000/N_BATCHES)), 
                                                           last_epoch=-1)# lr will have fallen to factor GAMMA after ~100,000 training steps
        
        start0 = start1 = time.time(); w1 = get_weights(model)
        for ie in range(N_EPOCHS):  # loop over the dataset multiple times
            
            wait_start, wait_time, gpu_time, gpu_utilisation = time.time(), 0., 0., 0.
            for ib in range(N_BATCHES):
                i = ioffset + ie*N_BATCHES+ib
                
                try:# get next train sample_batch
                    sample_batch = next(trainloader_iterator)
                except StopIteration:# restart trainloader
                    del trainloader_iterator
                    trainloader_iterator = iter(trainloader)# re-initiates batch/sampler iterators, with new random starts
                    sample_batch = next(trainloader_iterator)
                    #print(sample_batch["i"])# check
                   
                wait_time += time.time()-wait_start
                
                
                ## TRAIN
                
                gpu_start = time.time()
                
                model.train()# switch to train mode (for dropout/ batch norm layers)
                
                # get the data
                inputs = sample_batch["inputs"]# expects list of inputs
                labels = sample_batch["labels"]# expects list of labels
                inputs = [inp.to(device) for inp in inputs]
                labels = [lab.to(device) for lab in labels]
                
                # zero the parameter gradients  AT EACH STEP
                optimizer.zero_grad()# zeros all parameter gradient buffers
        
                # forward + backward + optimize
                outputs = model(*inputs)# expect tuple of outputs
                loss = c.LOSS(*labels, *outputs, c)# note loss is on cuda if labels/ outputs on cuda
                loss.backward()# updates all gradients in model
                optimizer.step()# updates all parameters using their gradients
                
                gpu_time += time.time()-gpu_start
                
                ## TRAIN STATISTICS
                
                if (i + 1) % 100 == 0:
                    gpu_utilisation = 100*gpu_time/(wait_time+gpu_time)
                    print("Wait time average: %.4f s GPU time average: %.4f s GPU util: %.2f %% device: %i"%(wait_time/100, gpu_time/100, gpu_utilisation, c.DEVICE))
                    gpu_time, wait_time = 0.,0.
                    
                if (i + 1) % c.SUMMARY_FREQ == 0:
                    
                    rate = c.SUMMARY_FREQ/(time.time()-start1)
                    
                    with torch.no_grad():# faster inference without tracking
                        
                        model.eval()
                        
                        # get example outputs and losses
                        inputs = sample_batch["inputs"]# expects list of inputs
                        labels = sample_batch["labels"]# expects list of labels
                        inputs = [inp.to(device) for inp in inputs]
                        labels = [lab.to(device) for lab in labels]
                        outputs = model(*inputs)
                        
                        l1loss = losses.l1_mean_loss(labels[0], outputs[0]).item()
                        l2loss = losses.l2_mean_loss(labels[0], outputs[0]).item()
                        kldloss = losses.kld_mean_loss(*labels, *outputs, c)
                        
                        writer.add_scalar("loss/l1_loss/train", l1loss, i + 1)
                        writer.add_scalar("loss/l2_loss/train", l2loss, i + 1)
                        writer.add_scalar("loss/kld_loss/train", kldloss, i + 1)
                        
                        if (i + 1) % (50 * c.SUMMARY_FREQ) == 0:
                            inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
                            outputs_array = outputs[0].detach().cpu().numpy().copy()
                            labels_array = labels[0].detach().cpu().numpy().copy()
                            f = plot_result(inputs_array, outputs_array, labels_array, model, c)
                            writer.add_figure("compare/train", f, i + 1, close=True)
                        
                        # check weight updates from previous summary
                        w2 = get_weights(model)
                        mu, _, av = get_weights_update_percent(w1, w2)
                        s = "Weight updates (%.1f %% average): "%(100*av)
                        for m in mu: s+="%.1f "%(100*m)
                        print(s)
                        del w1; w1 = w2

                        # add run statistics
                        writer.add_scalar("stats/epoch", ie, i + 1)
                        writer.add_scalar("stats/rate/batch", rate, i + 1)
                        writer.add_scalar("stats/rate/gpu_utilisation", gpu_utilisation, i + 1)
                        
                        # output to screen
                        print('[epoch: %i/%i, batch: %i/%i i: %i] l2loss: %.4f rate: %.1f elapsed: %.2f hr %s %s' % (
                               ie + 1,
                               N_EPOCHS,
                               ib + 1, 
                               N_BATCHES, 
                               i + 1,
                               l2loss,
                               rate,
                               (time.time()-start0)/(60*60),
                               time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()),
                               c.RUN
                                ))
                        
                    start1 = time.time()
                
                ## TEST STATISTICS
                
                if (i + 1) % c.TEST_FREQ == 0:
                    
                    with torch.no_grad():# faster inference without tracking
                        
                        try:# get next test sample_batch
                            sample_batch = next(testloader_iterator)
                        except StopIteration:# restart testloader
                            del testloader_iterator
                            testloader_iterator = iter(testloader)# re-initiates batch/sampler iterators, with new random starts
                            sample_batch = next(testloader_iterator)
                            #print(sample_batch["i"])# check
                        
                        model.eval()
                        
                        # get example outputs and losses
                        inputs = sample_batch["inputs"]# expects list of inputs
                        labels = sample_batch["labels"]# expects list of labels
                        inputs = [inp.to(device) for inp in inputs]
                        labels = [lab.to(device) for lab in labels]
                        outputs = model(*inputs)
                        
                        l1loss = losses.l1_mean_loss(labels[0], outputs[0]).item()
                        l2loss = losses.l2_mean_loss(labels[0], outputs[0]).item()
                        kldloss = losses.kld_mean_loss(*labels, *outputs, c)
                        
                        writer.add_scalar("loss/l1_loss/test", l1loss, i + 1)
                        writer.add_scalar("loss/l2_loss/test", l2loss, i + 1)
                        writer.add_scalar("loss/kld_loss/test", kldloss, i + 1)
                        
                        if (i + 1) % (50 * c.TEST_FREQ) == 0:
                            inputs_array = inputs[0].detach().cpu().numpy().copy()# detach returns a new tensor, detached from the current graph
                            outputs_array = outputs[0].detach().cpu().numpy().copy()
                            labels_array = labels[0].detach().cpu().numpy().copy()
                            f = plot_result(inputs_array, outputs_array, labels_array, model, c)
                            writer.add_figure("compare/test", f, i + 1, close=True)
                
                ## SAVE
                
                if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                    
                    model.eval()
                    
                    model.to(torch.device('cpu'))# put model on cpu before saving
                    # to avoid out-of-memory error
                    
                    # save a checkpoint
                    torch.save({
                    'i': i + 1,
                    'model_state_dict': model.state_dict(),
                    }, c.MODEL_OUT_DIR+"model_%.8i.torch"%(i + 1))
                    
                    model.to(device)
    
                wait_start = time.time()
                
            # AFTER EACH EPOCH
            scheduler.step()
            print('[epoch: %i/%i %i] learning rate adjusted: %s  (%s)'%(# output to screen
                   ie + 1,
                   N_EPOCHS,
                   i + 1,
                   scheduler.get_lr(), optimizer.param_groups[0]['lr']))
            writer.add_scalar("x_lr", optimizer.param_groups[0]['lr'], i + 1)
            
        del trainloader_iterator, testloader_iterator
            
        print('Finished Training (total runtime: %.1f hrs)'%(
                        (time.time()-start0)/(60*60)))
        
    def close(self):
        self.writer.close()
         

if __name__ == "__main__":
    
    #cs = [Constants()]
    
    DEVICE = 3


    cs = []
    
    '''
    #for threshold in [500, 1000]:
    #for threshold in [2000, 4000,]:
    #for threshold in [8000, 16000]:
    for threshold in [32000,]:
        
        for beta in [0.3,0.2,0.1,0.5]:
            
            data_path = "../data_c7_processed/Xf_7_%i.npy"%(threshold)
            n_examples = np.load(data_path).shape[0]
            
            cs.append(Constants(RUN="vae_final10_4_t%i_b%.2f_all"%(threshold,beta),
                                N_LATENT=4,
                                DATA_PATH=data_path,
                                N_EXAMPLES=n_examples,
                                DEVICE=DEVICE,
                                VAE_BETA=beta,
                                ))
    '''
    
    # FINAL RUN
    
    cs = []
    
    #for n_latent in [4,1]:
    #for n_latent in [2,5]:
    #for n_latent in [3,6]:
    for n_latent in [7,]:
        
    #for n_latent in [4,]:
        
        #for i in range(0,10):
        #for i in range(10,20):
        #for i in range(20,30):
        #for i in range(30,40):
            
            cs.append(Constants(RUN="vae_final11_%i"%(n_latent),
                                N_LATENT=n_latent,
                                DEVICE=DEVICE,
                                ))
        

    
    for c in cs:
        run = Trainer(c)
        run.train()
        run.close()
    