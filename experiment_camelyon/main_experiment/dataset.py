import cv2
import matplotlib.pyplot as plt
import numpy as np

import random

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter




def row_idx2img_path(row_idx, metadata ,base_dir):
    row = metadata.iloc[row_idx:row_idx+1,:]
    
    subfolder_dir = ['patient',
                     ''.join(['0'*(3 - len(str(row.iloc[0]['patient']))),str(row.iloc[0]['patient'])]), # 004
                     'node',str(row.iloc[0]['node'])]  # ['patient', '004', 'node', '4']
    
    subfolder_dir = '_'.join(subfolder_dir) # patient_004_node_4

    img_name = ['patch',
           subfolder_dir,
        'x', str(row.iloc[0]['x_coord']),
        'y', str(row.iloc[0]['y_coord'])] # 'patch', 'patient_004_node_4', 'x', '3328', 'y', '21792']

    img_name = '_'.join(img_name) # patch_patient_004_node_4_x_3328_y_21792

    img_name = img_name+'.png' # patch_patient_004_node_4_x_3328_y_21792.png

    img_path = '/'.join([base_dir, subfolder_dir, img_name]) 
    # './data/camelyon17_v1.0/patches/patient_004_node_4/patch_patient_004_node_4_x_3328_y_21792.png'
    
    return img_path


class camelyonDataset(Dataset):

    def __init__(self, metadata, base_dir):
        """
        Args:
            metadata (data frame): pandas dataframe of metadata
            base_dir (string): path to the base directory
        """
        self.metadata = metadata
        self.base_dir = base_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = row_idx2img_path(idx, metadata = self.metadata,
                                   base_dir = self.base_dir)
        image = cv2.imread(img_name)
        
        labels = self.metadata.iloc[idx,:]['tumor']
        dat = {'image': image, 'label': labels}


        return dat



def row_idx2info(row_idx,# a list of indices from metadata
                 metadata,# metadata.csv in the c dataset
                 base_dir,# base directory of images
                 info # a list of columns names in metadata; 'path' can be included
                ):
    output =  pd.DataFrame(dict({'indices':row_idx}),index = row_idx)
    if 'path' in info:
        paths = []
        for idx in row_idx:
            row = metadata.iloc[idx,]
            subfolder_dir = ['patient',
                         ''.join(['0'*(3 - len(str(row['patient']))),str(row['patient'])]), # 004
                         'node',str(row['node'])]  # ['patient', '004', 'node', '4']

            subfolder_dir = '_'.join(subfolder_dir) # patient_004_node_4

            img_name = ['patch',
                   subfolder_dir,
                'x', str(row['x_coord']),
                'y', str(row['y_coord'])] # 'patch', 'patient_004_node_4', 'x', '3328', 'y', '21792']

            img_name = '_'.join(img_name) # patch_patient_004_node_4_x_3328_y_21792

            img_name = img_name+'.png' # patch_patient_004_node_4_x_3328_y_21792.png

            img_path = '/'.join([base_dir, subfolder_dir, img_name]) 
            paths.append(img_path)
        if len(info) != 1:
            info.remove('path')
            output = pd.concat([output,metadata[info].iloc[row_idx,:]],axis = 1)
            output['path'] = paths

        else:
           # output = pd.DataFrame(dict({'path':paths}))
            output['path'] = paths
    else:
        #output = metadata[info].iloc[row_idx,:]
        output = pd.concat([output,metadata[info].iloc[row_idx,:]],axis = 1)
    output.index =range(0,output.shape[0])
    return(output)


def cluster_features(K, # int, number of clusters
                     which_target,# which hospital is the target, int from 0 to 4
                     seed,# random state form kmeans clustering on source data
                     row_idx,# list of row indices from the metadata
                     metadata,# metadata, subset of the original 'metadata.csv' inherented in the dataset
                     base_dir,
                     pca_50_features
                    ):
    # get information of examples with certain indices
    sub_meta = row_idx2info(row_idx, metadata, base_dir,info = ['center','tumor'])

    # get indices in `sub_meta`. number should within the range of 0 to nrow(sub_meta)-1
    target_idx = sub_meta.index[sub_meta['center'] == which_target].to_list()
    source_idx = sub_meta.index[sub_meta['center'] != which_target].to_list()

    if pca_50_features is None:
        source_meta = pd.concat([ sub_meta.iloc[source_idx,:],
           pd.DataFrame(dict({'cluster' : sub_meta.iloc[source_idx,:]['center']}),
                       index = sub_meta.index[source_idx]
                       )],axis = 1)
        target_meta = pd.concat([ sub_meta.iloc[target_idx,:],
           pd.DataFrame(dict({'cluster' : 'Target'}),
                       index = sub_meta.index[target_idx]
                       )],axis = 1)
    else: 
        # cluster source_feature
        # standardize
        source_feature = pca_50_features.iloc[source_idx,:]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(source_feature)
        # fit
        kmeans = KMeans(init="random",
                n_clusters=K,
                random_state=seed
               )
        kmeans.fit(scaled_features)

        # get the meta information of source data. The index are from 0 to nrow(sub_meta)-1. the column 'index' are from 0 to nrow(metadata)-1
        source_meta = pd.concat([ sub_meta.iloc[source_idx,:],
           pd.DataFrame(dict({'cluster' : kmeans.labels_}),
                       index = sub_meta.index[source_idx]
                       )],axis = 1)
        # get the meta information of target data where the 'cluster' column is all specified as 'Target'
        target_meta = pd.concat([ sub_meta.iloc[target_idx,:],
           pd.DataFrame(dict({'cluster' : 'Target'}),
                       index = sub_meta.index[target_idx]
                       )],axis = 1)

    return(pd.concat([source_meta,target_meta],axis = 0))



def check_balance(indices,metadata,base_dir):
    balance = row_idx2info(indices ,
                 metadata = metadata,# metadata.csv in the c dataset
                 base_dir = base_dir,# base directory of images
                 info = ['tumor'] # a list of columns names in metadata; 'path' can be included
                )['tumor'].pipe(Counter)
    print(balance)



def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()





def split_source(K, # positive integer: number of clusters
                 split_criteria, # ['center','cluster','random']
                 row_idx, # a list of int, row indices of data in use (in metadata)
                 metadata,# metadata in camelyon dataset
                 base_dir, # base directory of metadata
                 which_target = None, # an integer or list of row indices. split is on the rest of data
                 feature = None, # needs to be a pd df if split_criteria = 'cluster'
                 seed = 0 # random seed for clustering (if split_criteria = 'cluster')
                    ):
    # get information of examples with certain indices
    sub_meta = row_idx2info(row_idx, metadata, base_dir,info = ['center','tumor'])
    
    # get indices of target / source in `sub_meta`.
    # !number should within the range of 0 to nrow(sub_meta)-1
    if which_target is None:
        source_idx = sub_meta.index.to_list()
        
        target_meta = None
    elif type(which_target )== int:
        target_idx = sub_meta.index[sub_meta['center'] == which_target].to_list()
        source_idx = sub_meta.index[sub_meta['center'] != which_target].to_list()
        
        target_meta = pd.concat([ sub_meta.iloc[target_idx,:],
           pd.DataFrame(dict({'cluster' : 'Target'}),
                       index = sub_meta.index[target_idx]
                       )],axis = 1)
        
        
    elif type(which_target) == list:
        target_idx = sub_meta.loc[sub_meta['indices'].isin( which_target)].index.to_list()
        source_idx = sub_meta.loc[sub_meta['indices'].isin( which_target) == False].index.to_list()
        
        target_meta = pd.concat([ sub_meta.iloc[target_idx,:],
           pd.DataFrame(dict({'cluster' : 'Target'}),
                       index = sub_meta.index[target_idx]
                       )],axis = 1)
        
    
    if split_criteria == 'center':
        # data frame, column "cluster" is the same as senter
        source_meta = pd.concat([ sub_meta.iloc[source_idx,:],
           pd.DataFrame(dict({'cluster' : sub_meta.iloc[source_idx,:]['center']}),
                       index = sub_meta.index[source_idx]
                       )],axis = 1)
        
    elif split_criteria == 'cluster':
        try:
            # cluster source_feature
            # standardize
            source_feature = feature.iloc[source_idx,:]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(source_feature)
            # fit
            kmeans = KMeans(init="random",
                    n_clusters=K,
                    random_state=seed
                   )
            kmeans.fit(scaled_features)

            # get the meta information of source data. The index are from 0 to nrow(sub_meta)-1. the column 'index' are from 0 to nrow(metadata)-1
            source_meta = pd.concat([ sub_meta.iloc[source_idx,:],
               pd.DataFrame(dict({'cluster' : kmeans.labels_}),
                           index = sub_meta.index[source_idx]
                           )],axis = 1)
            
        except NameError:
            print('Please set the argument feature')
            return None

        
    elif split_criteria == 'random':
        random.seed(seed)
        clusters = random.choices(population=range(0,K),k=len(source_idx))
        source_meta = pd.concat([ sub_meta.iloc[source_idx,:],
               pd.DataFrame(dict({'cluster' : clusters}),
                           index = sub_meta.index[source_idx]
                           )],axis = 1)

    if which_target is None:
        return source_meta
    else:
        return pd.concat([source_meta,target_meta],axis = 0)
