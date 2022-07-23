from dataset import *
from train import *

import os
import pandas as pd

import torch
import torch.optim as optim

# Set parameters
EuroSat_Type = 'ALL'    # use 'RGB' or 'ALL' for type of Eurosat Dataset. Just change in this line. Rest of the code is managed for both type
target_country = 'United Kingdom'
lr = 0.01               # learn_rate
milestones = [50, 75, 90] # multistep scheduler
epochs =  100           # no of epochs
output_path = "./" + target_country

# raw data
if EuroSat_Type == 'RGB':
  data_folder = '/content/sample_data/'
elif EuroSat_Type == 'ALL':
    root = 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif/'
    download_ON = os.path.exists(root)
    if not download_ON:
      os.system('wget http://madm.dfki.de/files/sentinel/EuroSATallBands.zip') #All bands
      os.system('unzip EuroSATallBands.zip')
      download_ON = True

    
data = torchvision.datasets.DatasetFolder(root=root,loader = iloader, transform=None, extensions = 'tif')


# Metadata
geo_df = pd.read_csv("metadata.csv")
geo_dict = geo_df.to_dict()
countries = list(set(geo_dict["country"].values()))
id_countries = dict.fromkeys(countries)
for k in id_countries.keys():
    id_countries[k] = [v for (i, v) in enumerate(geo_dict["id"]) if geo_dict["country"][i] == k]


# source - target split
id_target = id_countries[target_country]
id_train = random.sample(id_target, 640)
id_test = list(set(id_target) - set(id_train))#[0:160]
id_test = random.sample(id_test, 160)

loader_target_train = torch.utils.data.DataLoader(torch.utils.data.Subset(data, id_train), 
                                                  batch_size = 16, shuffle = True, num_workers = 0)
loader_target_test = torch.utils.data.DataLoader(torch.utils.data.Subset(data, id_test), 
                                                  batch_size = 16, shuffle = True, num_workers = 0)

id_random_source = random.sample(list(geo_dict["id"].values()),
                                len(loader_target_train.dataset))
loader_random_source = torch.utils.data.DataLoader(torch.utils.data.Subset(data, id_random_source), 
                                                  batch_size= 16, shuffle=False, num_workers=0)

## Train
np.random.seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
print("pytorch version", torch.__version__)
criteria = torch.nn.CrossEntropyLoss()
net = Load_model()
if torch.cuda.is_available():
    print("cuda is available")
    net=net.cuda()
else:
    print("cuda is not available")
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

net_random = Load_model()
net_random = train(net_random, loader_random_source, loader_target_test, criteria, optimizer, epochs, scheduler)
torch.save(net_random.state_dict(), output_path + "random_source.pt" )

net = train(net, loader_target_train, loader_target_test, criteria, optimizer, epochs, scheduler)
torch.save(net.state_dict(), output_path + "target_train.pt" )

