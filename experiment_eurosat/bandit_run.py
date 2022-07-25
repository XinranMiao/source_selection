from dataset import *
from train import *

import os
import pandas as pd

import collections

import sys

import torch
import torch.optim as optim

# Set parameters
EuroSat_Type = 'ALL'    
args = sys.argv
target_task = str(args[1])
algorithm = str(args[2])
print("target is ", target_task, ", algorithm is ", algorithm, "\n")

from pathlib import Path
output_path = Path("derived_data")
output_path.mkdir(parents = True, exist_ok = True)


# Download data
if EuroSat_Type == 'RGB':
  data_folder = '/content/sample_data/'
  #root = os.path.join(data_folder, '2750/')
  root = '2750/'
  download_ON = os.path.exists(root)

  if not download_ON:
    # This can be long...
    #os.chdir(data_folder)
    os.system('wget http://madm.dfki.de/files/sentinel/EuroSAT.zip') #Just RGB Bands
    os.system('unzip EuroSA.zip')
    download_ON = True
elif EuroSat_Type == 'ALL':
    root = 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/'
    download_ON = os.path.exists(root)
    if not download_ON:
      os.system('wget http://madm.dfki.de/files/sentinel/EuroSATallBands.zip') #All bands
      os.system('unzip EuroSATallBands.zip')
      download_ON = True
geo_df = pd.read_csv("metadata.csv")

# Load data
data = torchvision.datasets.DatasetFolder(root=root,loader = iloader, transform=None, extensions = 'tif')
input_data = prepare_input_data(geo_df, target_task)


# Set seed
np.random.seed(0)
torch.cuda.manual_seed(0)
random.seed(0)

# Run model
_, bandit_selects, accs, alpha, beta, pi = bandit_selection(data, input_data, 
                                                            n_epochs = 50, n_it = 100,
                                                            algorithm = algorithm, iter_samples = 160,
                                                           output_path = output_path)

