#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:39:53 2018

@author: bmoseley
"""

import numpy as np


get_weights = lambda model: [p.detach().cpu().numpy().copy() for p in model.parameters()]

def get_weights_update_percent(weights1, weights2):
    assert len(weights1) == len(weights2)
    
    N = sum([w.size for w in weights1])
    epsilon = 1e-12
    
    mean, std, sum_all = [],[], 0
    for i in range(len(weights1)):
        w1, w2 = weights1[i], weights2[i]
        d = np.abs((w2 - w1)/(np.mean(np.abs(w1)) + epsilon))
        mean.append(np.mean(d))
        std.append(np.std(d))
        sum_all += np.sum(d)
    return mean, std, sum_all/N