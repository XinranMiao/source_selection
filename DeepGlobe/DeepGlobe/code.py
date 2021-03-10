import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import  transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms.functional as TF

from tqdm import tqdm

import numpy as np
import random
from random import choice
import data
#from data import create_dir, download_data
import re
import os
import cv2 
import glob
import matplotlib
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import time

from train1 import *
from features import *
from unet import *
from data_deepglobe import *
args = {
    "batch_size": 8,
    "epochs": 50,
    "lr": 0.0001,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu') ,# set to "cuda" if GPU is available
    "n_components":2
}
base_dir = "land-"
paths = {
    "x": list(glob.glob(base_dir+"train/*sat*")),
    "y": list(glob.glob(base_dir+"train/*.png"))
}
ds = DeepGlobeDataset(x_paths = paths["x"], y_paths = paths["y"])
print('loaded dataset')
validation_split = .15
test_split = .15
validation_size = round(len(ds) * validation_split)
test_size = round(len(ds) * test_split)
train_size = len(ds) - validation_size - test_size

train_ds, validation_ds,test_ds = torch.utils.data.random_split(ds, [train_size, validation_size,test_size])
train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=True)
validation_loader = DataLoader(validation_ds, batch_size=args["batch_size"], shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args["batch_size"], shuffle=True)
print(len(train_loader.dataset), len(validation_loader.dataset),len(test_loader.dataset))

print('Splitted dataset')
vggmodel = models.vgg16(pretrained=True).to(args["device"])
print('loaded vggmodel')
#start_time = time.time()
#for i, d in enumerate(train_loader):
 #   x = d['image'].to(args["device"])
  #  feature = extract_features(x,model = vggmodel,device = args['device'],flatten = True)
    #if i == 0:
   #     x_features = feature
    #else:
     #   x_features = torch.cat((x_features, feature), 0)
    #print('i=',i,'x_features.shape = ',x_features.shape)
    #print("--- %s seconds ---" % (time.time() - start_time))
x_features = np.load('x_features_cropped.npy')
pca_2d = PCA(n_components=args['n_components'])
pca_2d.fit(x_features)
pca_2d_features = pca_2d.transform(x_features)
print(pca_2d_features.shape)
kmeans = KMeans(n_clusters=4)
kmeans.fit(pca_2d_features)
'kmeans done'
model = Unet(3, 7,2).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
print('start training')
#train_epoch(model, train_loader, optimizer, args['device'], epoch=10)
print('start validating')
l=validate(model, test_loader,args['device'])
print(l)
