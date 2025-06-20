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
from scene.dataset_readers_batch import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.cameras import PseudoCamera
from utils.pose_utils import generate_random_poses


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], indices=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.train_cameras = None
        self.test_cameras = None
        self.counter = 0
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, indices=indices)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []

            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.num_train_cameras = len(scene_info.train_cameras)
        self.num_test_cameras = len(scene_info.test_cameras)
        self.train_cameras = scene_info.train_cameras
        self.test_cameras = scene_info.test_cameras
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            if len(scene_info.train_cameras) < 200:
                pseudo_cams = []
                pseudo_poses = generate_random_poses(self.train_cameras[resolution_scale])

                view = self.train_cameras[resolution_scale][0]
                for pose in pseudo_poses:
                    pseudo_cams.append(PseudoCamera(
                        R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                        width=view.image_width, height=view.image_height
                    ))
                self.pseudo_cameras[resolution_scale] = pseudo_cams

        if self.loaded_iter == "continue":
            pass
        elif self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
    
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        # Create an empty list to store selected cameras
        selected_cameras = []

        # From self.counter, select batch elements
        for i in range(self.args.batch):
            # Use modulo to ensure the index wraps around if it exceeds the list size
            index = (self.counter + i) % len(self.train_cameras)
            selected_cameras.append(self.train_cameras[index])

        # Update the counter by moving forward batch positions
        self.counter = (self.counter + self.args.batch) % len(self.train_cameras)

        return cameraList_from_camInfos(selected_cameras, scale, self.args)

    def getTestCameras(self, scale=1.0):
        if len(self.test_cameras) == 0:
            return []
        # Create an empty list to store selected cameras
        selected_cameras = []

        # From self.counter, select batch elements
        for i in range(self.args.batch):
            # Use modulo to ensure the index wraps around if it exceeds the list size
            index = (self.counter + i) % len(self.test_cameras)
            selected_cameras.append(self.test_cameras[index])

        # Update the counter by moving forward batch positions
        self.counter = (self.counter + self.args.batch) % len(self.test_cameras)

        return cameraList_from_camInfos(selected_cameras, scale, self.args)
    
    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]