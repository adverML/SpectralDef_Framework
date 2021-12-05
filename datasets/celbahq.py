""" train and test dataset

author baiyu
"""
import os
import sys
import pdb

import pandas as pd 
from PIL import Image
# import pickle

# from skimage import io
import matplotlib.pyplot as plt
# import numpy
import torch
from torch.utils.data import Dataset


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, data='Gender', transform=None):

        print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)
    
        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = self.df.index.values
        self.y = self.df[data].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index])) # R, G, B
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.df.shape[0]


class CelebaDatasetPath(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, data='Gender', transform=None):
        
        print("csv_path: ", csv_path)
        print("img_dir: ", img_dir)

        self.df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = self.df.index.values
        self.y = self.df[data].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label, self.img_names[index], self.y

    def __len__(self):
        return self.df.shape[0]