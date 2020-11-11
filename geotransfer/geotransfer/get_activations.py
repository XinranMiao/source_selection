import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout,AdaptiveAvgPool2d
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torchvision import models

import time
from sklearn.metrics import accuracy_score
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []
def get_all_activations(model,layers,img):
    save_output = SaveOutput()
    hook_handles = []

    for layer in layers:
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
        
    out = model(img)

    return save_output.outputs