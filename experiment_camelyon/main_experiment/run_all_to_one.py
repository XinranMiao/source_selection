# import modules
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

from dataset import *
from ensemble_module_resnet import *


# read data
base_dir = './data/camelyon17_v1.0/patches'
dataset = get_dataset(dataset='camelyon17', download=True)
metadata = pd.read_csv('./data/camelyon17_v1.0/metadata.csv', index_col=0)

camelyon17_dataset = camelyonDataset(metadata = metadata,base_dir = './data/camelyon17_v1.0/patches')
input_dir = './processed_data/'
df_10k = pd.read_csv(input_dir + 'df_10k.csv')
df_2500 = pd.read_csv(input_dir + 'df_2500.csv')
df_7500 = pd.read_csv(input_dir + 'df_7500.csv')

pca_50d_features_10k = pd.read_csv(input_dir+'resnet_pca_50d_features_10k.csv',index_col = 0)
pca_50d_shallow_features_10k = pd.read_csv(input_dir+'shallow_resnet_pca_50d_features_10k.csv',index_col = 0)

# read argument
which_target = int(sys.argv[1])
K = int(sys.argv[2])
seed = int(sys.argv[3])
split_criteria = str(sys.argv[4])
feature = str(sys.argv[5])
if feature == 'DeepResnetPca':
    feature_in_use = pca_50d_features_10k
elif feature == 'ShallowResnetPca':
    feature_in_use = pca_50d_shallow_features_10k
    
print('target is ',str(which_target+1),',seed = ',str(seed),
     'K = ',str(K+1-1),',split criteria = ',split_criteria,'\n')
print(feature_in_use.head())

batch_size = 16
n_epochs = 30
# for bandit
T = 100# number of rounds
step = 30
# ensemble 
n_training = 3000
n =30
# create directories from output
output_base_dir = '0926_results_'
output_dir = output_base_dir + 'target' + str(which_target)+ "_K"+str(K)+'_seed'+str(seed)+'_'+split_criteria + '_' + feature+'/'
#if os.path.isdir(output_base_dir)==False:
 #   os.mkdir(output_base_dir)
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)


df_clusters = split_source(K=K, # positive integer: number of clusters
                 split_criteria=split_criteria, # ['center','cluster','random']
                 row_idx=df_7500['indices'].to_list(), # a list of int, row indices of data in use (in metadata)
                 metadata=metadata,# metadata in camelyon dataset
                 base_dir = base_dir, # base directory of metadata
                 which_target = None, # an integer or list of row indices. split is on the rest of data
                 feature = feature_in_use, # needs to be a pd df if split_criteria = 'cluster'
                 seed = seed # random seed for clustering (if split_criteria = 'cluster')
                    )
df_clusters.to_csv(output_dir+'df_clusters.csv')

test_sample = df_2500.loc[df_2500['center']==which_target,'indices'].to_list()

train_sample = df_7500['indices'].to_list()
test_subset = torch.utils.data.Subset(camelyon17_dataset,test_sample)
test_subset_loader = torch.utils.data.DataLoader(test_subset, 
                                                  batch_size=batch_size)
train_subset = torch.utils.data.Subset(camelyon17_dataset,train_sample)
train_subset_loader = torch.utils.data.DataLoader(train_subset, 
                                                  batch_size=batch_size)

test_y_true = row_idx2info(test_sample,
                 metadata = metadata,# metadata.csv in the c dataset
                 base_dir = base_dir,# base directory of images
                 info  = ['tumor']# a list of columns names in metadata; 'path' can be included
                )['tumor'].to_list()

check_balance(test_sample,metadata = metadata,base_dir = base_dir)
check_balance(train_sample,metadata = metadata, base_dir = base_dir)

idx_cluster = df_clusters.loc[df_clusters['cluster'] != 'Target',['indices','cluster']]
unique_clusters =list(set(idx_cluster['cluster']))

n_cluster = np.array(unique_clusters).max() + 1
myDict = {key: [] for key in unique_clusters}
for i in range(idx_cluster.shape[0]):
    for clust in range(n_cluster):
        if (idx_cluster.iloc[i,1] == clust) or (idx_cluster.iloc[i,1] == str(clust)):
            myDict[clust].append(idx_cluster.iloc[i,0])

if 'df_source_balance' in vars():
    del df_source_balance
for i in myDict.keys():
    if 'df_source_balance' not in vars():
        df_source_balance = pd.DataFrame(Counter(metadata.iloc[myDict[i],:]['tumor']),index = [i]
                                        )
    else:
        df_source_balance = df_source_balance.append(pd.DataFrame(Counter(metadata.iloc[myDict[i],:]['tumor']),
                                                                  index =[ i]
                                                                 )
                                                    )
df_source_balance.to_csv(output_dir+'df_source_balance.csv')   


# Ensemble ##############
#from ensemble_module_resnet import *
print('ENSEMBLE START')
train_samples,test_results,dat = generate_dat(n_training = n_training,
                                             n = n,
                                              dataset = camelyon17_dataset,
                                              myDict = myDict,
                                              batch_size = batch_size,
                                             K = K,
                                             test_subset_loader = test_subset_loader,
                                             n_epochs =n_epochs)

np.savetxt(output_dir+"_nsemble_test_results.csv", 
           test_results.reshape(test_results.shape[0], -1))
pd.DataFrame(dat).to_csv(output_dir+'ensemble_dat.csv')

pd.DataFrame(train_samples).to_csv(output_dir+'ensemble_train_samples.csv')




# Bandit ##############
print('BANDIT START')
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = Linear(in_features = 512, out_features = 2,bias = True)

# specify loss function (categorical cross-entropy)
criterion = CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

alpha = np.repeat(1,K)
beta = np.repeat(1,K)

Choice_clusters = []
Y = []
train_Accuracy = []
val_Accuracy = []
Accuracy = []

train_sample = []
test_predictions = np.empty((T,len(test_sample)))

for t in range(T):
    # draw samples from the Beta distribution for parameters
    pi = []
    for i in range(K):
        pi.append(np.random.beta(alpha[i],beta[i]))
    c = np.where(pi == np.array(pi).max());c = np.asarray(c) 
    # If there are multiple clusters with the largest prior, we'll randomly choose one.
    maxidx = choice(c[0,:])
    Choice_clusters.append(maxidx)
    
    train_sample_t =random.choices(myDict[maxidx],k=step)
    train_sample = train_sample+train_sample_t
    
    # split the training source data further into train and test
    train_subset = torch.utils.data.Subset(camelyon17_dataset,train_sample)
    train_subset_train, train_subset_val = torch.utils.data.random_split(dataset = train_subset,
                                                                         lengths=[round(0.8 * len(train_subset)),
                                                                                                    len(train_subset) - round(0.8* len(train_subset))])
    val_y_true = row_idx2info(train_subset_val.indices,
                          metadata = metadata,# metadata.csv in the c dataset
                 base_dir = base_dir,# base directory of images
                 info  = ['tumor']# a list of columns names in metadata; 'path' can be included
                )['tumor'].to_list()
    train_subset_train_loader = torch.utils.data.DataLoader(train_subset_train,
                                                  batch_size=batch_size)

    train_subset_val_loader = torch.utils.data.DataLoader(train_subset_val,
                                                  batch_size=batch_size)# train the model
    l,a = train_model(train_subset_train_loader,model,criterion,optimizer,n_epochs= n_epochs,batch_size = batch_size)
   
    # evaluate
    with torch.no_grad():
        val_acc,val_pred,val_prob =  pred_acc(train_subset_val_loader,model,batch_size=batch_size)
        test_acc,test_pred,test_prob = pred_acc(test_subset_loader,model,batch_size=batch_size)
    test_predictions[t,:] = test_pred
    print('t = ',t,'cluster = ',maxidx,'train acc = ',np.average(a),
          ', val acc = ',np.average(val_acc),
          ', test acc = ',np.average(test_acc),'\n')
    Accuracy.append(np.average(test_acc))
    train_Accuracy.append(np.average(np.array(a)))
    val_Accuracy.append(np.average(np.array(val_acc)))
    
    
    # print some info
    
    # training sample balance
    train_balance = row_idx2info(train_sample ,
                 metadata = metadata,# metadata.csv in the c dataset
                 base_dir = base_dir,# base directory of images
                 info = ['tumor'] # a list of columns names in metadata; 'path' can be included
                )['tumor'].pipe(Counter)
    print('The training sample balance is ',train_balance)

    
    # update parameters
    if np.average(test_acc) > Accuracy[max(t-1,0)]:
        Y.append(1)
        alpha[Choice_clusters[t]] = alpha[Choice_clusters[t]]+1
    else:
        Y.append(0)
        beta[Choice_clusters[t]] = beta[Choice_clusters[t]] + 1




pd.DataFrame({
    'Y':Y,
    'Choice_clusters':Choice_clusters,
    'Accuracy':Accuracy
}).to_csv(output_dir+'/bandit_T_sample_1.csv')

pd.DataFrame(test_predictions).to_csv(output_dir+'bandit_test_predictions_sample_1.csv')
pd.DataFrame(train_sample).to_csv(output_dir+'bandit_train_sample_sample_1.csv')

torch.save(model.state_dict(), output_dir+'bandit_model.pt')

