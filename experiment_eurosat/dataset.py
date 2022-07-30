import random
import time

import numpy as np

# geo
from affine import Affine
from geopy.geocoders import Nominatim
from pyproj import Proj, transform
import rasterio

from skimage.transform import resize
from sklearn.model_selection import train_test_split

from skimage import io

import torchvision
import torch


def iloader(path):
    image = np.asarray((io.imread(path))/32000,dtype='float32')
    return image.transpose(2,0,1)


def Load_data(root, EuroSat_Type = "ALL"):
    if EuroSat_Type == 'RGB':
      data = torchvision.datasets.DatasetFolder(root=root,loader = iloader, transform=None, extensions = 'jpg')
    elif EuroSat_Type == 'ALL':
      data = torchvision.datasets.DatasetFolder(root=root,loader = iloader, transform=None, extensions = 'tif')
    train_set, val_set = train_test_split(data, test_size=0.2, stratify=data.targets)
    val_set, test_set = train_test_split(data, test_size=0.01, stratify=data.targets)
    #print(np.unique(train_set, return_counts=True))  #uncomment for class IDs
    #print(np.unique(val_set, return_counts=True))    #uncomment for class IDs
      
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, drop_last = True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=16, shuffle=True, num_workers=0, drop_last = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0, drop_last = True)
    return train_loader, val_loader ,test_loader



def get_coords(fname):
    # Read raster
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        p1 = Proj(r.crs)
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows[0,0], cols[0,0])

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    longs, lats = transform(p1, p2, eastings, northings)
    return longs, lats

def locate(fname = None, long = None, lat = None):
    if not fname is None:
        long, lat = get_coords(fname)
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.reverse(str(lat)+","+str(long))
    return location.raw['address']



# Augmentation
def get_random_pos(img, window_shape = [55,55] ):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    #x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    #y2 = y1 + h
    return x1, x1 + w, y1, y1 + h #x1, x2, y1, y2

def random_crop_area(img):
    x1,x2,y1,y2 = get_random_pos(img)
    Sen_Im = img[:, x1:x2,y1:y2]
    return resize(Sen_Im,img.shape,anti_aliasing=True)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cus_aug(data):
    data = torch.rot90(data,random.randint(-3,3), dims=random.choice([[3,2],[2,3]]))
    if random.random()>0.75:
        data = torch.flip(data, dims = random.choice([[2,],[3,],[2,3]]))
    pixmis = torch.empty_like(data).random_(data.shape[-1])
    pixmis = torch.where(pixmis>(data.shape[-1]/8),torch.ones_like(data),torch.zeros_like(data))
    return data* pixmis

def check_labels(input_data, labels):
    """
    check labels across source / target train / target validation / target test sets, and keep labels in common
    """
    train_labels = [labels[i] for i in input_data["idx_train"]]
    val_labels = [labels[i] for i in input_data["idx_val"]]
    test_labels = [labels[i] for i in input_data["idx_test"]]
    source_labels = [labels[i] for i in input_data["idx_source"]]

    common_labels = list(set(train_labels).intersection(val_labels).intersection(test_labels).intersection(source_labels))

    input_data["idx_train"] = [i for i in input_data["idx_train"] if labels[i] in common_labels]
    input_data["idx_test"] = [i for i in input_data["idx_test"] if labels[i] in common_labels]
    input_data["idx_val"] = [i for i in input_data["idx_val"] if labels[i] in common_labels]
    input_data["idx_source"] = [i for i in input_data["idx_source"] if labels[i] in common_labels]
    
    return input_data

    
def prepare_input_data(geo_df, target_task, labels = None, 
                       train_size = .6, test_size = .5, target_size = 1600):
    
    geo_dict = geo_df.to_dict()
    countries = list(set(geo_dict["country"].values()))
    countries = [x for x in countries if str(x) != "nan"]
    id_countries = dict.fromkeys(countries)
    for k in id_countries.keys():
        id_countries[k] = [v for (i, v) in enumerate(geo_dict["id"]) if geo_dict["country"][i] == k]

    # create a dictionary for input data
    
    input_data = {
        "source_task": list(set(id_countries.keys()) - set([target_task])),
        "target_task": target_task
    }

    
    # all data, both source and target
    
    input_data["data_dict"] = {}
    for k in geo_dict.keys():
        input_data["data_dict"][k] = [geo_dict[k][i] for (i, v) in enumerate(geo_dict["country"].values()) if str(v) != "nan"]


        
    # split indices for source and target
    
    input_data["idx_source"] = [i for (i, v) in enumerate(input_data["data_dict"]['country']) if v != input_data["target_task"]]
    input_data["idx_target"] = [i for (i, v) in enumerate(input_data["data_dict"]['country']) if v == input_data["target_task"]]

    target_labels = list(set([labels[i] for i in input_data["idx_target"]]))

    
    # For source data, create a dictionary to record the countries
    
    input_data["source_dict"] = {}
    for k in geo_dict.keys():
        input_data["source_dict"][k] = [input_data["data_dict"][k][i] for i in input_data["idx_source"] if labels[i] in target_labels]

   
    # resample the target to make the number of samples is fixed
    
    if len(input_data["idx_target"]) >= target_size:
        input_data["idx_target"] = random.sample(input_data["idx_target"], k = target_size)
    else:
        input_data["idx_target"] = random.choices(input_data["idx_target"], k = target_size)
        
    
    # split the target data into train / validation / test sets
    
    y_target = [labels[i] for i in input_data["idx_target"]]
    input_data["idx_train"], idx_rest, _, y_rest = train_test_split(input_data["idx_target"],
                                                              y_target,
                                                              test_size = 1 - train_size,
                                                              random_state = 0, shuffle = True)
    input_data["idx_val"], input_data["idx_test"], _, _ = train_test_split(idx_rest,
                                                              y_rest,
                                                              test_size = test_size,
                                                              random_state = 0, shuffle = True)

    return input_data

