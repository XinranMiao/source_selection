
from torch.utils.data import Dataset, DataLoader
from torchvision import  transforms
import glob
import numpy as np
import os
import random
#import shutil
#import tarfile
import torch
import torchvision.transforms.functional as TF
#import urllib.request
import cv2 
from skimage.util.shape import view_as_windows

def slice_tile(img, size=(512, 512), overlap=0):
    """Slice an image into overlapping patches
    Args:
        img (np.array): image to be sliced
        size tuple(int, int, int): size of the slices
        overlap (int): how much the slices should overlap
    Returns:
        list of slices [np.array]"""
    size_ = (size[0], size[1], img.shape[2])
    patches = view_as_windows(img, size_, step=size[0] - overlap)
    result = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            result.append(patches[i, j, 0])
    return result

def slice_pair(img, mask, **kwargs):
    """
    Slice an image / mask pair
    """
    # maskout areas with nans
    nan_mask = np.isnan(img[:, :, 0])
    nan_mask = np.expand_dims(nan_mask, axis=2)
    nan_mask = np.repeat(nan_mask, mask.shape[-1], axis=2)
    mask[nan_mask] = 0

    img_slices = slice_tile(img, **kwargs)
    mask_slices = slice_tile(mask, **kwargs)
    return img_slices, mask_slices

class DeepGlobeDataset_old(Dataset):
    def __init__(self, x_paths, y_paths, imsize=2448):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.imsize = imsize
        self.mapping =  {(0,255,255):0,
                         (255,255,0):1,
                         (255,0,0):4,
                         (255,0,255):2,
                         (0,255,0):3,
                         #(0,0,255):4,
                         (255,255,255):5,
                         (0,0,0):6}

    def preprocess(self,img):
        #w, h, c = img.shape
        #newW, newH = int(scale * w), int(scale * h)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans
            
    def RGB2classes(self,mask,h=2448,w=2448,c=7):
        mask = torch.from_numpy(mask)
        mask = torch.squeeze(mask)
        mask = mask.permute(2,0,1).contiguous()
        
        mask_classes = np.zeros((h,w,c)) # mask[0,62,1], mask[0,0,:]

        for k in self.mapping:
            idx = (mask== torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))   
            validx = (idx.sum(0) == 3)          
            mask_classes[:,:,self.mapping[k]][validx] = torch.tensor(1, dtype=torch.long)
        return mask_classes

    def __getitem__(self, index):
        x_path = self.x_paths[index]
        y_path = self.y_paths[index]
        
        x_img = cv2.imread(x_path)
        y_img = cv2.imread(y_path)
        
        x_img = self.preprocess(x_img)
        
        y_img = self.RGB2classes(y_img)
        y_img = self.preprocess(y_img)
        
        return {
            'mask': y_img,#torch.from_numpy(y_img).type(torch.FloatTensor),
            'image': torch.from_numpy(x_img).type(torch.FloatTensor)
        }


    def __len__(self):
        return len(self.x_paths)

        
class DeepGlobeDataset(Dataset):
    def __init__(self, x_paths, y_paths, imsize=2448):
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.imsize = imsize
        self.mapping =  {(0,255,255):0,
                         (255,255,0):1,
                         (255,0,0):4,
                         (255,0,255):2,
                         (0,255,0):3,
                         #(0,0,255):4,
                         (255,255,255):5,
                         (0,0,0):6}

    def preprocess(self,img):
        #w, h, c = img.shape
        #newW, newH = int(scale * w), int(scale * h)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans
            
    def RGB2classes(self,mask,h=2448,w=2448,c=7):
        mask = torch.from_numpy(mask)
        mask = torch.squeeze(mask)
        mask = mask.permute(2,0,1).contiguous()
        
        mask_classes = np.zeros((h,w,c)) # mask[0,62,1], mask[0,0,:]

        for k in self.mapping:
            idx = (mask== torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))   
            validx = (idx.sum(0) == 3)          
            mask_classes[:,:,self.mapping[k]][validx] = torch.tensor(1, dtype=torch.long)
        return mask_classes

    def __getitem__(self, index):
        n = round((index +1)/16) - 1
        m = (index+1)%16 - 1
        
        x_path = self.x_paths[n]
        y_path = self.y_paths[n]
        
        x_img = cv2.imread(x_path)
        y_img = cv2.imread(y_path)
        x_img,y_img = slice_pair(x_img,y_img, size=(512, 512), overlap=0)
        
        x_img = x_img[m]
        y_img = y_img[m]
        
        x_img = self.preprocess(x_img)
        
        y_img = self.RGB2classes(y_img,h = 512, w = 512)
        y_img = self.preprocess(y_img)
        
        
        
        
        
        return {
            'mask': y_img,#torch.from_numpy(y_img).type(torch.FloatTensor),
            'image': torch.from_numpy(x_img).type(torch.FloatTensor)
        }


    def __len__(self):
        return len(self.x_paths * 16)

