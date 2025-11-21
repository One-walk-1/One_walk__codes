#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import random

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm
from einops import rearrange
from utils.dataset_utils import split_dataset_by_rx_man, split_dataset_by_rx_frame


def split_dataset(datadir, ratio=0.1, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrum')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')

def split_dataset_man(datadir, ratio=0.1, dataset_type='rfid'):
    """random shuffle train/test set
    """
    if dataset_type == "rfid":
        spectrum_dir = os.path.join(datadir, 'spectrums')
        spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
        index = [x.split('.')[0] for x in spt_names]
        random.shuffle(index)

    train_len = int(len(index) * ratio)
    print(train_len)
    train_index = np.array(index[:train_len])
    test_index = np.array(index[train_len:])

    np.savetxt(os.path.join(datadir, "train_index.txt"), train_index, fmt='%s')
    np.savetxt(os.path.join(datadir, "test_index.txt"), test_index, fmt='%s')

def split_dataset_rx_man(datadir, ratio=0.8, dataset_type='rfid', seed=42, position_precision=None):
    assert 0 < ratio < 1, "ratio should be between 0 and 1"
    split_dataset_by_rx_man(
        datadir=datadir,
        rx_ratio=ratio,
        man_ratio=ratio,
        seed=seed,
        position_precision=position_precision,
        verbose=False
    )    

class Spectrum_dataset(Dataset):
    """Spectrum dataset class."""
    
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir  
        self.rx_pos_dir = os.path.join(datadir, 'rx_info.csv')  
        self.spectrum_dir = os.path.join(datadir, 'spectrum')  
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])       
        self.dataset_index = np.loadtxt(indexdir, dtype=str)  
        # self.rx_pos = pd.read_csv(self.rx_pos_dir).values  
        rx_info = pd.read_csv(self.rx_pos_dir)
        # self.orientations = rx_info[['qx', 'qy', 'qz', 'qw']].values
        self.positions = rx_info[['x', 'y', 'z']].values
        self.n_samples = len(self.dataset_index)  

    def __len__(self):
        return self.n_samples 

    def __getitem__(self, index):
        
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0  
        spectrum = torch.tensor(spectrum, dtype=torch.float32)  

        # rx_pos_i = torch.tensor(self.rx_pos[int(self.dataset_index[index]) - 1], dtype=torch.float32)
        idx = int(self.dataset_index[index]) - 1  
        # orientation_i = torch.tensor(self.orientations[idx], dtype=torch.float32)
        position_i = torch.tensor(self.positions[idx], dtype=torch.float32)
        return spectrum, position_i # , orientation_i  

class SimpleRxDataset(Dataset):
    def __init__(self, datadir) -> None:
        super().__init__()
        self.datadir = datadir
        self.rx_info_file = os.path.join(datadir, "rx_info.csv")   
        self.gt_dir = os.path.join(datadir, "spectrum")                 

        rx_info = pd.read_csv(self.rx_info_file)

        self.rx_pos = rx_info[['x', 'y', 'z']].iloc[0].values.astype(np.float32)

        self.orientations = rx_info[['qx', 'qy', 'qz', 'qw']].values.astype(np.float32)

        self.gt_names = sorted([f for f in os.listdir(self.gt_dir) if f.endswith(".png")])
        assert len(self.orientations) == len(self.gt_names), "not match between orientations and ground_truth images"

        self.n_samples = len(self.orientations)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        rx_pos_i = torch.tensor(self.rx_pos, dtype=torch.float32)
        orientation_i = torch.tensor(self.orientations[index], dtype=torch.float32)

        img_path = os.path.join(self.gt_dir, self.gt_names[index])
        gt_img = imageio.imread(img_path) / 255.0
        gt_img = torch.tensor(gt_img, dtype=torch.float32)

        return rx_pos_i, orientation_i, gt_img

class Spectrum_man_dataset(Dataset):
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir  
        self.spectrum_info_dir = os.path.join(datadir, 'spectrum_info.csv')  
        self.spectrum_dir = os.path.join(datadir, 'spectrums')  
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])       
        self.dataset_index = np.loadtxt(indexdir, dtype=str)  
        # self.rx_pos = pd.read_csv(self.rx_pos_dir).values  
        spectrum_info = pd.read_csv(self.spectrum_info_dir)
        self.index = spectrum_info['index'].values
        self.rx_pos = spectrum_info[['rx_x', 'rx_y', 'rx_z']].values
        self.man_pos = spectrum_info[['man_x', 'man_y', 'man_z']].values
        self.man_orient = spectrum_info[['qx', 'qy', 'qz', 'qw']].values
        self.n_samples = len(self.dataset_index)

    def __len__(self):
        return self.n_samples 

    def __getitem__(self, index):
        
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0  
        spectrum = torch.tensor(spectrum, dtype=torch.float32)  

        # rx_pos_i = torch.tensor(self.rx_pos[int(self.dataset_index[index]) - 1], dtype=torch.float32)
        idx = int(self.dataset_index[index]) - 1  
        rx_pos_i = torch.tensor(self.rx_pos[idx], dtype=torch.float32)
        man_pos_i = torch.tensor(self.man_pos[idx], dtype=torch.float32)
        man_orient_i = torch.tensor(self.man_orient[idx], dtype=torch.float32)
        return spectrum, rx_pos_i, man_pos_i, man_orient_i

class Spectrum_id_dataset(Dataset):
    def __init__(self, datadir, indexdir) -> None:
        super().__init__()
        self.datadir = datadir  
        self.spectrum_info_dir = os.path.join(datadir, 'spectrum_info.csv')  
        self.spectrum_dir = os.path.join(datadir, 'spectrums')  
        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])       
        self.dataset_index = np.loadtxt(indexdir, dtype=str)  
        # self.rx_pos = pd.read_csv(self.rx_pos_dir).values  
        spectrum_info = pd.read_csv(self.spectrum_info_dir)
        self.index = spectrum_info['index'].values
        self.rx_pos = spectrum_info[['rx_x', 'rx_y', 'rx_z']].values
        # self.man_pos = spectrum_info[['man_x', 'man_y', 'man_z']].values
        # self.man_orient = spectrum_info[['qx', 'qy', 'qz', 'qw']].values
        self.frame_ids = spectrum_info['frame_idx'].values
        self.n_samples = len(self.dataset_index)

    def __len__(self):
        return self.n_samples 

    def __getitem__(self, index):
        
        img_name = os.path.join(self.spectrum_dir, self.dataset_index[index] + '.png')
        spectrum = imageio.imread(img_name) / 255.0  
        spectrum = torch.tensor(spectrum, dtype=torch.float32)  

        # rx_pos_i = torch.tensor(self.rx_pos[int(self.dataset_index[index]) - 1], dtype=torch.float32)
        idx = int(self.dataset_index[index]) - 1  
        rx_pos_i = torch.tensor(self.rx_pos[idx], dtype=torch.float32)
        # man_pos_i = torch.tensor(self.man_pos[idx], dtype=torch.float32)
        # man_orient_i = torch.tensor(self.man_orient[idx], dtype=torch.float32)
        frame_id_i = torch.tensor(self.frame_ids[idx], dtype=torch.long)
        return spectrum, rx_pos_i, frame_id_i

dataset_dict = {"rfid": Spectrum_dataset, "localization": SimpleRxDataset, 
                "man": Spectrum_man_dataset, "id": Spectrum_id_dataset}
