import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tools import read_write_model
from math import atan

def focal2fov(focal, pixels):
    return 2 * atan(pixels / (2 * focal))

def load_pose(pose_txt):
    pose = []
    with open(pose_txt, 'r') as f:
        for line in f:
            row = line.strip('\n').split()
            row = [float(c) for c in row]
            pose.append(row)
    pose = np.array(pose).astype(np.float32)
    assert pose.shape == (4,4)
    return pose

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]

        image = np.array(Image.open(color_path))
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path)) / self.depth_scale

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


class v2_360_Dataset(MonocularDataset):
    def __init__(self, args, path, config, data_folder, scene):
        super().__init__(args, path, config)
        self.has_depth = False
        self.v2_360_Parser(data_folder, scene) 
        
    def v2_360_Parser(self, data_folder, scene):
        self.color_paths, self.poses, self.depth_paths = [], [], []

        gt_dirs = Path(data_folder) / scene / "sparse/0"
        _, images, _ = read_write_model.read_model(gt_dirs, ".bin")

        test_images = Path(data_folder) / scene / "train_views/triangulated/list_test.txt"
        name2id = {image.name: i for i, image in images.items()}

        with open(test_images, "r") as f:
            while True:
                image_name = f.readline()
                if not image_name:
                    break
                image_name = image_name.strip()
                if len(image_name) > 0 and image_name[0] != "#":
                    image_path = Path(data_folder) / scene / f"images_4/{image_name}"
                    self.color_paths.append(image_path)

                    image = images[name2id[image_name]]
                    R_gt, t_gt = image.qvec2rotmat(), image.tvec
                    pose = np.eye(4)            
                    pose[:3, :3] = R_gt         
                    pose[:3, 3] = t_gt 
                    self.poses.append(pose)


