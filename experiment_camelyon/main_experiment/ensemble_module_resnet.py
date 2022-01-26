
from collections import Counter
import numpy as np

import pandas as pd

import random
from random import choice

import sklearn
from sklearn.metrics import accuracy_score
import sys
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torchvision import models
from torch.nn import CrossEntropyLoss, Linear, Sequential
import torch.nn.functional as nnf

from typing import Type, Any, Callable, Union, List, Optional



from wilds import get_dataset

from resnet_modified import BasicBlock,ResNet


#subset_acts = torch.from_numpy(np.load('./input_data_0817/subset_acts.npy'))

#subset_acts = torch.from_numpy(np.load('./0726/subset_acts.npy'))

def train_model(loader, model, criterion, optimizer, n_epochs,batch_size):

    accuracy = [];losses = []
    probabilities = []
    train_since = time.time()

    for e in range(n_epochs):
        training_loss = []
        training_acc = []
        i = 0
        print('Epoch {}/{}'.format(e, n_epochs - 1))
        for d in loader:
            x = d['image']; x= x.swapaxes(2,3).swapaxes(1,2);x = x.float()
            y_true = d['label']
            optimizer.zero_grad()
            model = model.float()
        
            #f = model.features[42](subset_acts[(batch_size * i) : min(batch_size*i+batch_size, len(loader.dataset) ),:,:,:])
           # avgpool = model.avgpool(f)
            y_hat = model(x)

            prob = nnf.softmax(y_hat, dim=1)
            top_p, predictions = prob.topk(1, dim = 1)  
            predictions = predictions[:,0].numpy().tolist() 
            
            
            acc = accuracy_score(predictions,y_true)
        
            training_acc.append(acc)
            loss = criterion(y_hat.requires_grad_(True),y_true.type(torch.LongTensor))    
        
            training_loss.append(loss.item())
            loss.backward() # compute gradients of all variables wrt loss
            optimizer.step()# perform updates using calculated gradients
        
            i += 1
        accuracy.append(np.average(training_acc))
        losses.append(np.average(training_loss))   
        print('Loss: {:.4f} Acc: {:.4f}'.format(np.average(training_loss), np.average(training_acc)))
    train_time_elapsed = time.time() - train_since
    print('Training complete in {:.0f}m {:.0f}s'.format(train_time_elapsed // 60, train_time_elapsed % 60),
         'average acc is ',np.average(accuracy))
    return(losses,accuracy) # training loss / accuracy over epochs


def pred_acc(loader,model,batch_size):
    model.eval()
    since = time.time()
    # prediction for test set
    accuracy = []
    prediction = []
    y_true = []
    i = 0
    probabilities = []
    for d in loader:
        x = d['image']; x = x.swapaxes(2,3).swapaxes(1,2); x = x.float()
        
        output = model(x)
        prob = nnf.softmax(output, dim=1)
        top_p, predictions = prob.topk(1, dim = 1)  
        predictions = predictions[:,0].numpy().tolist()           
    
        i+=1
        if len(prediction) ==0:
            prediction = predictions
            y_true = d['label'].numpy().tolist()
            probabilities = prob
        else:
            prediction = prediction + predictions
            y_true = y_true + d['label'].numpy().tolist()
            probabilities = torch.cat((probabilities,prob),0)

    for j in range(len(prediction)):
            accuracy.append(accuracy_score([y_true[j]],[prediction[j]]) 
                   )
    time_elapsed = time.time() - since
    print('Predicting complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return accuracy,prediction,probabilities








def smlt(n_training,
        weights,
        myDict,
        test_subset_loader,
         n_epochs,batch_size,
         dataset,
        save_model = ''):
    keys_order = np.argsort(list(myDict.keys()))
    sorted_keys =list(myDict.keys())
    sorted_keys.sort()
    train_sample = []
    for i in keys_order:
        train_sample = train_sample + random.sample(myDict[sorted_keys[i]],
                                                    min(int(weights[i] * n_training),
                                                        len(myDict[sorted_keys[i]])))
    # initialize the model
    model = ResNet(block = BasicBlock,layers = [2, 2, 2, 2])
    model.fc = Linear(in_features = 512, out_features = 2,bias = True)
    model.load_state_dict(torch.load('processed_data/resnet_model.pt'),strict=False)
    #model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    #model.fc = Linear(in_features = 512, out_features = 2,bias = True)
    #for param in model.parameters():param.requires_grad = False
    #for param in model.classifier[6].parameters():param.requires_grad = True

    criterion = CrossEntropyLoss()

    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
    
    # set the data loader
    #train_subset_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,train_sample),
              #                                    batch_size=batch_size)
    train_subset = torch.utils.data.Subset(dataset,train_sample)
    train_subset_train, train_subset_val = torch.utils.data.random_split(dataset = train_subset,
                                                                         lengths=[round(0.8 * len(train_subset)),
                                                                                                    len(train_subset) - round(0.8* len(train_subset))])
    train_subset_train_loader = torch.utils.data.DataLoader(train_subset_train,
                                                  batch_size=batch_size)

    train_subset_val_loader = torch.utils.data.DataLoader(train_subset_val,
                                                  batch_size=batch_size)# train the model
    
    # training
    l,a = train_model(train_subset_train_loader, model, criterion, optimizer, n_epochs,batch_size)
    
    # evaluation
    with torch.no_grad():
        val_acc,val_pred,val_prob =  pred_acc(train_subset_val_loader,
                                                                     model,batch_size=batch_size)
        test_acc,test_pred,test_prob =  pred_acc(test_subset_loader,
                                                                     model,batch_size=batch_size)
    
    return train_sample,np.average(np.array(a)),np.average(np.array(val_acc)),test_acc,test_pred,test_prob[:,0].detach().numpy().tolist(),weights




def generate_dat(n_training,
                n,# number of simulations
                K, # number of clusters,
                test_subset_loader,
                dataset,
                n_epochs ,
                batch_size,
                myDict):
    train_samples = np.empty((n,n_training+2))
    test_results = np.empty((len(test_subset_loader.dataset),K,n))
    train_accs = []
    val_accs = []
    dat = np.empty((n,K+5))
    for i in range(n):
        
        print('-----The ',i,'-th simulation----')
        train_sample,train_acc,val_acc,test_acc,test_pred,test_prob0,weights = smlt(n_training =n_training,
                                                       myDict=myDict,
                                                      test_subset_loader=test_subset_loader ,
                                                      n_epochs=n_epochs,batch_size=batch_size,
                                                      dataset = dataset  ,
                                                      weights= np.random.dirichlet(np.repeat(1,K)))
        
     
    
        train_samples[i,0:len(train_sample)] = train_sample
        dat[i,0:K] = weights;dat[i,K:(K+1)] = np.average(test_acc) 
        print('The validation acc is ',val_acc,', test acc is ',np.average(test_acc))
        test_results[:,0,i] = test_acc;test_results[:,1,i] = test_pred;test_results[:,2,i] = test_prob0
        train_accs.append(train_acc)
        val_accs.append(val_acc)
    return train_samples,test_results,dat






