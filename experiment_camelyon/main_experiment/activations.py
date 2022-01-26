from collections import Counter


import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import os 
import pandas as pd

import random
from random import choice


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
from torch.nn import CrossEntropyLoss, Linear, Sequential
import torch.nn.functional as nnf
from wilds import get_dataset

def idx2acts(indices, # list
            metadata,
             dataset,
            model = 'resnet'):
    ind_acts =indices
    for i in ind_acts:
        if i == ind_acts[0]:
            subset_x =  torch.tensor([dataset[i]['image']],
                          dtype = torch.uint8)
        else:
            subset_x  = torch.cat((subset_x,
                      torch.tensor([dataset[i]['image']])),
                      0)
    subset_x = subset_x.swapaxes(2,3).swapaxes(1,2).float() 
    return subset_x


    
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
def activations(model, layers, x, device=None):
    """Get all activation vectors over images for a model.
    :param model: A pytorch model
    :type model: currently is Net defined by ourselves
    :param layers: One or more layers that activations are desired
    :type layers: torch.nn.modules.container.Sequential
    :param x: A 4-d tensor containing the test datapoints from which activations are desired.
                The 1st dimension should be the number of test datapoints.
                The next 3 dimensions should match the input of the model
    :type x: torch.Tensor
    :param device: A torch.device, specifying whether to put the input to cpu or gpu.
    :type device: torch.device
    :return (output): A list containing activations of all specified layers.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_output = SaveOutput()
    hook_handles = []

    for layer in layers:
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

    with torch.no_grad():
      x = x.to(device)
      out = model(x)

    output = save_output.outputs.copy()
    del handle,save_output, hook_handles, out
    torch.cuda.empty_cache()
    return output
def prepare_acts_input(loader,ind):
    x = torch.tensor(np.expand_dims(loader.dataset[ind]['image'],0))
    x= x.swapaxes(2,3).swapaxes(1,2);x = x.float()
    return x
def plot_acts(acts, # a list of activations
            # model ,#
              rows =8,
              columns = 8
             ):
    fig = plt.figure(figsize=(20, 20)) 
    for i in range(acts.shape[1]):
        fig.add_subplot(rows, columns, i+1) 
        plt.imshow(acts[0,i,:,:], cmap='gray'#, vmin=-1, vmax
        ) 
        plt.axis('off')
