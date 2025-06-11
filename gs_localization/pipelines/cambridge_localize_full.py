import os
import h5py
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import sys
import yaml
from munch import munchify
from math import atan
from collections import OrderedDict

sys.path.append("D:/gs-localization/gaussian_splatting")
sys.path.append("D:/gs-localization")
sys.path.append("D:/gs-localization/gs_localization/pipelines")

from tools.config_utils import load_config, update_recursive
from tools import read_write_model
from tools.gaussian_model import GaussianModel
from tools import render
from tools.camera_utils import Camera
from tools.descent_utils import get_loss_tracking
from tools.pose_utils import update_pose
from tools.graphics_utils import getProjectionMatrix2

def gradient_decent(viewpoint, config, initial_R, initial_T):

    viewpoint.update_RT(initial_R, initial_T)
    
    opt_params = []
    opt_params.append(
        {
            "params": [viewpoint.cam_rot_delta],
            "lr": 0.0001,
            "name": "rot_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.cam_trans_delta],
            "lr": 0.001,
            "name": "trans_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_a],
            "lr": 0.001,
            "name": "exposure_a_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_b],
            "lr": 0.001,
            "name": "exposure_b_{}".format(viewpoint.uid),
        }
    )
    

    pose_optimizer = torch.optim.Adam(opt_params)
    
    for tracking_itr in range(20):
        
        render_pkg = render(
            viewpoint, Model, pipeline_params, background
        )
        
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
          
        pose_optimizer.zero_grad()
        
        loss_tracking = get_loss_tracking(
            config, image, depth, opacity, viewpoint
        )
        loss_tracking.backward()
        
    
        with torch.no_grad():
            pose_optimizer.step()
            converged = update_pose(viewpoint, converged_threshold=1e-3)
    
        if converged:
            break
             
    return viewpoint.R, viewpoint.T, render_pkg


class Transformation:
    def __init__(self, R=None, T=None):
        self.R = R
        self.T = T

def quat_to_rotmat(qvec):
    qvec = np.array(qvec, dtype=float)
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R


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

def create_mask(mkpts_lst, width, height, k):
    # Initial mask as all False
    mask = np.zeros((height, width), dtype=bool)
    
    # Calculate k radius
    half_k = k // 2
    
    # Iterate through all points
    for pt in mkpts_lst:
        x, y = int(pt[0]), int(pt[1])
        
        # Calculate k*k borders
        x_min = max(0, x - half_k)
        x_max = min(width, x + half_k + 1)
        y_min = max(0, y - half_k)
        y_max = min(height, y + half_k + 1)
        
        # Set mask k*k area as True
        mask[y_min:y_max, x_min:x_max] = True
    
    # Shape: (1, height, width)
    mask = mask[np.newaxis, :, :]
    
    return mask

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass

class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]

        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None


    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        intrinsic = self.intrinsics[idx]

        image = np.array(Image.open(color_path))
        depth = None

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
        intrinsic = torch.from_numpy(intrinsic).to(device=self.device)
        projection_matrix = self.projection_matrices[idx].to(device="cuda:0")
        
        return image, depth, pose, intrinsic, projection_matrix 

def parse_camera_params(file_path):
    camera_params = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            
            # Ensure there are at least 8 elements, and the second element is 'PINHOLE'
            if len(parts) == 8 and parts[1] == 'PINHOLE':
                try:
                    # Extract parameters, skipping the second item 'PINHOLE'
                    img_name = parts[0]
                    w = int(parts[2])
                    h = int(parts[3])
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    
                    # Store in the dictionary
                    camera_params[img_name] = {
                        'fx': fx,
                        'fy': fy,
                        'w': w,
                        'h': h,
                        'cx': cx,
                        'cy': cy
                    }
                except ValueError as e:
                    print(f"Error parsing line: {line}. Error: {e}")
            else:
                print(f"Line skipped due to incorrect format: {line}")
    
    return camera_params
    
class cambridge_Dataset(MonocularDataset):
    def __init__(self, args, path, config, data_folder, scene):
        super().__init__(args, path, config)
        self.has_depth = False
        self.cambridge_Parser(data_folder, scene) 
        
    def cambridge_Parser(self, data_folder, scene):
        self.color_paths, self.poses, self.depth_paths, self.intrinsics, self.projection_matrices = [], [], [], [], []

        gt_dirs = Path(data_folder) / f"CambridgeLandmarks_Colmap_Retriangulated_1024px/{scene}/empty_all"
        _, images, _ = read_write_model.read_model(gt_dirs, ".txt")
        output_folder = data_folder.replace("datasets", "output")
        
        # Read the filenames from test_fewshot.txt and store them in a set.
        test_images_path = Path(data_folder) / scene / "test_full.txt"
        
        with open(test_images_path, 'r') as f:
            test_images = set(line.strip() for line in f)

        intrinsics = parse_camera_params(f"D:/gs-localization/output/cambridge_full/{scene}/query_list_with_intrinsics.txt")

        for i, image in tqdm(images.items(),"Load dataset"):
            image_name = image.name.replace("/","_")
            # Execute the following operation only if image.name exists in test_images."
            if image_name in test_images:
                image_path = Path(data_folder) / scene / 'images_full' / image_name
                self.color_paths.append(image_path)
                R_gt, t_gt = image.qvec2rotmat(), image.tvec
                pose = np.eye(4)            
                pose[:3, :3] = R_gt         
                pose[:3, 3] = t_gt 
                self.poses.append(pose)
                self.depth_paths.append(None)
                
                intrinsic = intrinsics[image_name]     
                projection_matrix = getProjectionMatrix2(
                    znear=0.01,
                    zfar=100.0,
                    fx=intrinsic["fx"],
                    fy=intrinsic["fy"],
                    cx=intrinsic["cx"],
                    cy=intrinsic["cy"],
                    W=intrinsic["w"],
                    H=intrinsic["h"],
                ).transpose(0, 1)
    
                self.intrinsics.append(np.array([
                                       intrinsic["fx"],
                                       intrinsic["fy"],
                                       intrinsic["cx"],
                                       intrinsic["cy"],
                                       focal2fov(intrinsic["fx"], intrinsic["w"]),
                                       focal2fov(intrinsic["fy"], intrinsic["h"]),
                                       intrinsic["h"], 
                                       intrinsic["w"]
                                        ]))
                
                self.projection_matrices.append(projection_matrix)


        # Sort self.color_paths, self.poses, and self.depth_paths based on normal file name order
        sorted_data = sorted(zip(self.color_paths, self.depth_paths, self.poses, 
                                 self.intrinsics, self.projection_matrices), key=lambda x: x[0].name)
        self.color_paths, self.depth_paths, self.poses, self.intrinsics, self.projection_matrices = zip(*sorted_data)
        del images

with open("D:/gs-localization/gs_localization/pipelines/configs/mono/tum/fr3_office.yaml", "r") as f:
    cfg_special = yaml.full_load(f)

inherit_from = "D:/gs-localization/gs_localization/pipelines/configs/mono/tum/base_config.yaml"

if inherit_from is not None:
    cfg = load_config(inherit_from)
else:
    cfg = dict()

# merge per dataset cfg. and main cfg.
data_folder = "D:/gs-localization/datasets/cambridge"
config = update_recursive(cfg, cfg_special)
config = cfg
config["Training"]["monocular"] = True
config["Training"]["opacity_threshold"] = 0.99
config["Training"]["edge_threshold"] = 1.1

#for scene in ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
for scene in ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']:
    try:
        Model = GaussianModel(3, config)
        Model.load_ply(f"D:/gs-localization/output/cambridge_full/{scene}/gs_map/iteration_30000/point_cloud.ply")
    except:
        Model = GaussianModel(1, config)
        Model.load_ply(f"D:/gs-localization/output/cambridge_full/{scene}/gs_map/iteration_30000/point_cloud.ply")
    
    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])
    data_folder = "D:/gs-localization/datasets/cambridge"
    dataset = cambridge_Dataset(model_params, model_params.source_path, config, data_folder, scene)
    bg_color = [0, 0, 0] 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # use OrderedDict to substitute defaultdict
    test_infos = OrderedDict()
    
    # suppose file open and read
    with open(f"D:/gs-localization/output/cambridge_full/{scene}/results_sparse.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            name = parts[0]
            qvec = list(map(float, parts[1:5]))
            tvec = list(map(float, parts[5:8]))

            R = quat_to_rotmat(qvec)
            T = np.array(tvec)
    
            # insert directly in OrderedDict
            test_infos[name] = Transformation(R=R, T=T)
    
    # sort OrderedDict according to name 
    test_infos = OrderedDict(sorted(test_infos.items(), key=lambda item: item[0]))
    
    rot_errors = []
    trans_errors = []
    
    file = h5py.File(f'D:/gs-localization/output/cambridge_full/{scene}/feats-superpoint-n4096-r1024.h5', 'r')
    
    for i, image in enumerate(tqdm(test_infos, desc="Localization")):
        viewpoint = Camera.init_from_intrinsic(dataset, i)

        viewpoint.compute_grad_mask(config)
        
        group = file[image] 
        keypoints = group['keypoints'][group['scores'][:]>0.2]  
        mask = create_mask(mkpts_lst=keypoints, width=1024, height=576, k=10)
        viewpoint.grad_mask = viewpoint.grad_mask | torch.tensor(mask).to("cuda:0")
    
        initial_R = torch.tensor(test_infos[image].R)
        initial_T = torch.tensor(test_infos[image].T).squeeze()
    
        rotation_matrix, translation_vector, render_pkg = gradient_decent(viewpoint, config, initial_R, initial_T)
        #rotation_matrix, translation_vector = initial_R, initial_T
    
        R_gt = viewpoint.R_gt.cpu().numpy()
        t_gt = viewpoint.T_gt.reshape(3,1).cpu().numpy()
        R = rotation_matrix.cpu().numpy()
        t = translation_vector.reshape(3,1).cpu().numpy()
        trans_error = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
        cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
        rot_error = np.rad2deg(np.abs(np.arccos(cos)))
        #print(image, rot_error, trans_error)
        rot_errors.append(rot_error)
        trans_errors.append(trans_error)

        torch.cuda.empty_cache()
    
    np.save(f"D:/gs-localization/output/cambridge_full/{scene}/rot_errors.npy", rot_errors)
    np.save(f"D:/gs-localization/output/cambridge_full/{scene}/trans_errors.npy", trans_errors)
    med_t = np.median(trans_errors)
    med_R = np.median(rot_errors)
    print( f"\nMedian errors for {scene}: {med_t:.3f}m, {med_R:.3f}deg")
    
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((np.array(trans_errors) < th_t) & (np.array(rot_errors) < th_R))
        print(f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%")
        
    file.close()
