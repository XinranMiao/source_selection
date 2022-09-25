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
target_task = int(args[1])
algorithm = str(args[2])
target_size = int(args[3])
print("target is ", str(target_task), ", algorithm is ", algorithm, ", target size", str(target_size), "\n")

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
geo_df = pd.read_csv("metadata_clustered.csv")

# load data
data = torchvision.datasets.DatasetFolder(root=root,loader = iloader, transform = None, extensions = 'tif')
labels = [v[1] for (i, v) in enumerate(data)]

input_data = prepare_input_data(geo_df, target_task, group_by = "cluster", 
                                labels = labels,
                               target_size = target_size)
# Set seed
np.random.seed(0)
torch.cuda.manual_seed(0)
random.seed(0)

# prepare data ---

target_val_loader =  torch.utils.data.DataLoader(torch.utils.data.Subset(data, input_data["idx_val"]), 
                                              batch_size = 16, shuffle = True, num_workers = 0)
target_train_loader =  torch.utils.data.DataLoader(torch.utils.data.Subset(data, input_data["idx_train"]), 
                                                  batch_size = 16, shuffle = True, num_workers = 0)
target_test_loader =  torch.utils.data.DataLoader(torch.utils.data.Subset(data, input_data["idx_test"]), 
                                                  batch_size = 16, shuffle = True, num_workers = 0)



# initialize hyperparameters ---

bandit_selects = [None]
alpha = dict.fromkeys(input_data["source_task"], [1])
beta = dict.fromkeys(input_data["source_task"], [1])
pi = dict.fromkeys(input_data["source_task"], [0])



# Run model
net, bandit_selects, accs, alpha, beta, pi = bandit_selection(data, input_data, 
                                                            n_epochs = 2, n_it = 2,
                                                            algorithm = algorithm, iter_samples = 160,
                                                           output_path = output_path)



# test performance
test_performance, y_test, yhat_test = validation(net, target_test_loader)
pd.DataFrame({"test_acc": test_performance,
             "algorithm": algorithm,
             "target_size": target_size,
             "target_task": target_task},
            index = [0]).to_csv(output_path / Path(str(target_task) + "_" + algorithm + "_" + str(target_size) + "_test_acc.csv"))
test_dict = {"y_test": y_test,
               "yhat_test": yhat_test}
pd.DataFrame.from_dict(test_dict).to_csv(output_path / Path(str(target_task) + "_" + algorithm + "_" + str(target_size) + "_test_pred.csv"))