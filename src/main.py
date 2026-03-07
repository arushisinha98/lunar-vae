#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 17:04:43 2018

@author: bmoseley
"""


import sys
import os
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from constants import Constants
import losses
from torch_utils import get_weights, get_weights_update_percent
from figures import plot_prediction, plot_result


class Trainer:
    "Generic model trainer class"
    
    def __init__(self, c):
        "Initialise torch, output directories, training dataset and model"
        
        ## INITIALISE
        
        # set seed
        if c.SEED == None: c.SEED = torch.initial_seed()
        else: torch.manual_seed(c.SEED) # likely independent of numpy
        np.random.seed(c.SEED)
                       
        # clear directories
        c.get_outdirs()
        c.save_constants_file() # saves torch seed too
        print(c)
        
        # set device / threads
        device = torch.device("cuda:%i"%(c.DEVICE) if torch.cuda.is_available() else "cpu")
        print("Device: %s"%(device))
        torch.backends.cudnn.benchmark = False#let cudnn find the best algorithm to use for your hardware (not good for dynamic nets)
        torch.set_num_threads(1)# for main inference
        
        print("Main thread ID: %i"%os.getpid())
        print("Number of CPU threads: ", torch.get_num_threads())
        print("Torch seed: ", torch.initial_seed())
        
        # initialise summary writer
        writer = SummaryWriter(c.SUMMARY_OUT_DIR)
        writer.add_text("constants", str(c).replace("\n","  \n")) # uses markdown

        ### DEFINE TRAIN/TEST DATASETS
        
        # split dataset 80:20
        irange = np.arange(0, c.N_EXAMPLES)
        np.random.shuffle(irange) # randomly shuffle the indicies (in place) before splitting. To get diversity in train/test split.
        traindataset = c.DATASET(c,
                                 irange=irange[0:(8*c.N_EXAMPLES//10)],
                                 verbose=True)
        testdataset = c.DATASET(c,
                                irange=irange[(8*c.N_EXAMPLES//10):c.N_EXAMPLES],
                                verbose=True)
        assert len(set(traindataset.irange).intersection(testdataset.irange)) == 0# make sure examples aren't shared!
        
        #### DEFINE MODEL
        
        model = c.MODEL(c)
        
        # Resume from latest checkpoint if available
        latest_ckpt = os.path.join(c.MODEL_OUT_DIR, "model_latest.torch")
        self.resume_state = None
        if c.MODEL_LOAD_PATH is not None:
            cp = torch.load(c.MODEL_LOAD_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(cp['model_state_dict'])
            ioffset = cp.get("i", 0)
            self.resume_state = cp
            print(f"Loaded model weights from: {c.MODEL_LOAD_PATH}")
        elif os.path.exists(latest_ckpt):
            cp = torch.load(latest_ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(cp['model_state_dict'])
            ioffset = cp.get("i", 0)
            self.resume_state = cp
            print(f"Resumed from latest checkpoint: {latest_ckpt}")
        else:
            ioffset = 0
        
        model.to(device)

        # print out parameters
        writer.add_graph(model, torch.zeros((1,)+c.VELOCITY_SHAPE))# write graph before placing on GPU
        print()
        print("Model: %s"%(model.name))
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters: %i"%(total_params))
        print("Total number of trainable parameters: %i"%(total_trainable_params))
        for p in model.parameters(): print(p.size(), p.numel())

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
                        batch_size=c.BATCH_SIZE, # shuffle=True, # reshuffles data at every epoch (RandomSampler)
                        sampler=RandomSampler(traindataset, replacement=True),# randomly sample with replacement
                        num_workers=4, # num_workers = spawns multiprocessing subprocess workers
                        drop_last=True, # so that each batch is complete
                        timeout=0)
        
        testloader = DataLoader(testdataset,
                        batch_size=c.BATCH_SIZE, # shuffle=True, # reshuffles data at every epoch (RandomSampler)
                        sampler=RandomSampler(testdataset, replacement=True),# randomly sample with replacement
                        num_workers=4, # num_workers = spawns multiprocessing subprocess workers
                        drop_last=True, # so that each batch is complete
                        timeout=0)
        
        testloader_iterator = iter(testloader)
        trainloader_iterator = iter(trainloader)
        assert len(trainloader_iterator) == N_BATCHES
        
        optimizer = torch.optim.Adam(model.parameters(), lr=c.LRATE, weight_decay=c.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=np.exp(np.log(c.SCHEDULER_GAMMA)/(100000/N_BATCHES)), last_epoch=-1)

        # Resume optimizer and scheduler state if available
        start_step = 0
        if self.resume_state is not None:
            if 'optimizer_state_dict' in self.resume_state:
                optimizer.load_state_dict(self.resume_state['optimizer_state_dict'])
            if 'scheduler_state_dict' in self.resume_state:
                scheduler.load_state_dict(self.resume_state['scheduler_state_dict'])
            start_step = self.resume_state.get('i', 0)

        start0 = start1 = time.time(); w1 = get_weights(model)
        for ie in range(N_EPOCHS): # loop over the dataset multiple times
            
            wait_start, wait_time, gpu_time, gpu_utilisation = time.time(), 0., 0., 0.
            for ib in range(N_BATCHES):
                i = ioffset + ie*N_BATCHES+ib
                if i < start_step:
                    continue  # skip already completed steps
                
                try: # get next train sample_batch
                    sample_batch = next(trainloader_iterator)
                except StopIteration:# restart trainloader
                    del trainloader_iterator
                    trainloader_iterator = iter(trainloader) # re-initiates batch/sampler iterators, with new random starts
                    sample_batch = next(trainloader_iterator)
                   
                wait_time += time.time()-wait_start
                
                
                ## TRAIN
                
                gpu_start = time.time()
                
                model.train() # switch to train mode (for dropout/ batch norm layers)
                
                # get the data
                inputs = sample_batch["inputs"] # expects list of inputs
                labels = sample_batch["labels"] # expects list of labels
                inputs = [inp.to(device) for inp in inputs]
                labels = [lab.to(device) for lab in labels]
                
                # zero the parameter gradients  AT EACH STEP
                optimizer.zero_grad()# zeros all parameter gradient buffers
        
                # forward + backward + optimize
                outputs = model(*inputs) # expect tuple of outputs
                loss = c.LOSS(*labels, *outputs, c) # note loss is on cuda if labels/ outputs on cuda
                loss.backward() # updates all gradients in model
                optimizer.step() # updates all parameters using their gradients
                
                gpu_time += time.time()-gpu_start
                
                ## TRAIN STATISTICS
                
                if (i + 1) % 100 == 0:
                    gpu_utilisation = 100*gpu_time/(wait_time+gpu_time)
                    print("Wait time average: %.4f s GPU time average: %.4f s GPU util: %.2f %% device: %i"%(wait_time/100, gpu_time/100, gpu_utilisation, c.DEVICE))
                    gpu_time, wait_time = 0.,0.
                    
                if (i + 1) % c.SUMMARY_FREQ == 0:
                    
                    rate = c.SUMMARY_FREQ/(time.time()-start1)
                    
                    with torch.no_grad(): # faster inference without tracking
                        
                        model.eval()
                        
                        # get example outputs and losses
                        inputs = sample_batch["inputs"] # expects list of inputs
                        labels = sample_batch["labels"] # expects list of labels
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
                            inputs_array = inputs[0].detach().cpu().numpy().copy() # detach returns a new tensor, detached from the current graph
                            outputs_array = outputs[0].detach().cpu().numpy().copy()
                            f1 = plot_prediction(inputs_array, outputs_array)
                            writer.add_figure("compare/train", f1, i + 1, close=True)
                            f2 = plot_result(model, c)
                            writer.add_figure("compare/train", f2, i + 1, close=True)
                        
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
                        
                        try: # get next test sample_batch
                            sample_batch = next(testloader_iterator)
                        except StopIteration: # restart testloader
                            del testloader_iterator
                            testloader_iterator = iter(testloader) # re-initiates batch/sampler iterators, with new random starts
                            sample_batch = next(testloader_iterator)
                            # print(sample_batch["i"]) # check
                        
                        model.eval()
                        
                        # get example outputs and losses
                        inputs = sample_batch["inputs"] # expects list of inputs
                        labels = sample_batch["labels"] # expects list of labels
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
                            f1 = plot_prediction(inputs_array, outputs_array)
                            writer.add_figure("compare/test", f1, i + 1, close=True)
                            f2 = plot_result(model, c)
                            writer.add_figure("compare/test", f2, i + 1, close=True)
                
                ## SAVE
                
                if (i + 1) % c.MODEL_SAVE_FREQ == 0:
                    model.eval()
                    # model.to(torch.device('cpu'))
                    # Save a numbered checkpoint
                    torch.save({
                        'i': i + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, c.MODEL_OUT_DIR+"model_%.8i.torch" % (i + 1))
                    # Always save a latest checkpoint for resuming
                    torch.save({
                        'i': i + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, os.path.join(c.MODEL_OUT_DIR, "model_latest.torch"))
                    # model.to(device)

            # Always save a latest checkpoint at the end of each epoch
            model.eval()
            # model.to(torch.device('cpu'))
            torch.save({
                'i': i + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(c.MODEL_OUT_DIR, "model_latest.torch"))
            # model.to(device)
    
            wait_start = time.time()
                
            # AFTER EACH EPOCH
            scheduler.step()
            print('[epoch: %i/%i %i] learning rate adjusted: %s  (%s)'%(# output to screen
                   ie + 1,
                   N_EPOCHS,
                   i + 1,
                   scheduler.get_last_lr()[0], optimizer.param_groups[0]['lr']))
            writer.add_scalar("x_lr", optimizer.param_groups[0]['lr'], i + 1)
            
        del trainloader_iterator, testloader_iterator
            
        print('Finished Training (total runtime: %.1f hrs)'%(
                        (time.time()-start0)/(60*60)))

        # Save Final Model
        model.eval()
        torch.save(model.state_dict(), os.path.join(c.MODEL_OUT_DIR, "model_final.torch"))
        
        # To Reload:
        # model = c.MODEL(c)
        # model.load_state_dict(torch.load("model_final.torch", map_location="cpu"))
        # model.eval()
        
    def close(self):
        self.writer.close()
         

if __name__ == "__main__":
    
    cs = []
    cs = [Constants()]
    
    for c in cs:
        run = Trainer(c)
        run.train()
        run.close()
    