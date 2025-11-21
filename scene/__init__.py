#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import DataLoader
from scene.dataloader import *
import yaml
from scene.deform_model import DeformModel

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.batch_size = 1
        self.datadir = "./data/static_dataset"
        self.cameras_extent = 2
        self.datadir_man = "./data/dynamic_dataset"
        
        yaml_file_path = os.path.join(self.datadir, 'gateway_info.yml')  
        with open(yaml_file_path, 'r') as file:
            tx_data = yaml.safe_load(file)
        self.fixed_tx_position = tx_data['gateway1']['position']
        self.tx_orientation = tx_data['gateway1']['orientation']
        
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
            
        self.train_cameras = {}
        self.test_cameras = {}

        dataset = dataset_dict["rfid"]
        dataset_man = dataset_dict["man"]
        
        train_index = os.path.join(self.datadir, "train_index.txt")
        test_index = os.path.join(self.datadir, "test_index.txt")
        
        train_index_man = os.path.join(self.datadir_man, "train_index.txt")
        test_index_man = os.path.join(self.datadir_man, "test_index.txt")
        val_index_man = os.path.join(self.datadir_man, "val_index.txt")
        
        if not os.path.exists(train_index) or not os.path.exists(test_index):
            split_dataset(self.datadir, ratio=0.8, dataset_type="rfid")
            # split_dataset_by_orientation(self.datadir, dataset_type="rfid")

        if not os.path.exists(train_index_man) or not os.path.exists(test_index_man) or not os.path.exists(val_index_man):
            # split_dataset_man(self.datadir_man, ratio=0.8, dataset_type="rfid")
            split_dataset_rx_man(self.datadir_man, ratio=0.9, dataset_type="rfid", seed=42, position_precision=None)

        self.train_set = dataset(self.datadir, train_index)
        self.test_set = dataset(self.datadir, test_index)
        
        self.train_iter = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.train_set_man = dataset_man(self.datadir_man, train_index_man)
        self.test_set_man = dataset_man(self.datadir_man, test_index_man)
        self.val_set_man = dataset_man(self.datadir_man, val_index_man)

        self.train_iter_man = DataLoader(self.train_set_man, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_iter_man = DataLoader(self.test_set_man, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.val_iter_man = DataLoader(self.val_set_man, batch_size=self.batch_size, shuffle=False, num_workers=0)



    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def dataset_init(self):
        self.train_iter_dataset = iter(self.train_iter)
        self.test_iter_dataset = iter(self.test_iter)
        self.train_iter_man_dataset = iter(self.train_iter_man)
        self.test_iter_man_dataset = iter(self.test_iter_man)
        self.val_iter_man_dataset = iter(self.val_iter_man)
        
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
