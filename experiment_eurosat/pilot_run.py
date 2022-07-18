from dataset import *
from train import *

import os
import pandas as pd

import torch
import torch.optim as optim

# Set parameters
EuroSat_Type = 'ALL'    # use 'RGB' or 'ALL' for type of Eurosat Dataset. Just change in this line. Rest of the code is managed for both type
lr = 0.01               # learn_rate
milestones = [50,75,90] # multistep scheduler
epochs = 3            # no of epochs


# raw data
if EuroSat_Type == 'RGB':
  data_folder = '/content/sample_data/'
  #root = os.path.join(data_folder, '2750/')
  root = '2750/'
  download_ON = os.path.exists(root)

  if not download_ON:
    # This can be long...
    #os.chdir(data_folder)
    os.system('wget http://madm.dfki.de/files/sentinel/EuroSAT.zip') #Just RGB Bands
    !unzip EuroSAT.zip
    download_ON = True
elif EuroSat_Type == 'ALL':
    root = 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/'
    download_ON = os.path.exists(root)
    if not download_ON:
      os.system('wget http://madm.dfki.de/files/sentinel/EuroSATallBands.zip') #All bands
      !unzip EuroSATallBands.zip
      download_ON = True
    
data = torchvision.datasets.DatasetFolder(root=root,loader = iloader, transform=None, extensions = 'tif')


# Metadata
geo_df = pd.read_csv("metadata.csv")
geo_dict = geo_df.to_dict()
countries = list(set(geo_dict["country"].values()))
id_countries = dict.fromkeys(countries)
for k in id_countries.keys():
    id_countries[k] = [v for (i, v) in enumerate(geo_dict["id"]) if geo_dict["country"][i] == k]

