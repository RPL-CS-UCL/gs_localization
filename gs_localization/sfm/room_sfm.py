import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm
import torch
import logging
import PIL.Image
import numpy as np

gs_localization_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(gs_localization_path))

from hloc import (
    extract_features,
    match_features,
    logger,
    triangulation,
    pairs_from_covisibility,
    pairs_from_retrieval,
    localize_sfm
)
from hloc.utils.sfm_utils import create_query_list_with_intrinsics
from hloc.utils.read_write_model import read_model, write_model, read_images_text, qvec2rotmat, qvec2rotmat, read_cameras_binary, read_cameras_text, read_images_binary,\
    read_images_text,read_model,write_model

logger = logging.getLogger(__name__)

def scene_coordinates(p2D, R_w2c, t_w2c, depth, camera):
    assert len(depth) == len(p2D)

    # Intrinsic
    K = np.array([[camera.params[0], 0, camera.params[2]],
                  [0, camera.params[1], camera.params[3]],
                  [0, 0, 1]])
    
    # Inverse of intrinsic
    K_inv = np.linalg.inv(K)
    
    # Transform 2D pixels to camera frame
    p2D_homogeneous = np.concatenate([p2D, np.ones((p2D.shape[0], 1))], axis=1)
    p2D_h = p2D_homogeneous @ K_inv.T

    # From camera frame to 3D-camera-frame
    p3D_c = p2D_h * depth[:, None]
    
    # From 3D-camera-frame to 3D-world-frame
    p3D_w = (p3D_c - t_w2c) @ R_w2c
    
    return p3D_w


def interpolate_depth(depth, kp):
    h, w = depth.shape
    kp = kp / np.array([[w - 1, h - 1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    depth = torch.from_numpy(depth)[None, None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(depth, kp, align_corners=True, mode="bilinear")[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        depth, kp, align_corners=True, mode="nearest"
    )[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    interp_depth = interp.T.numpy().flatten()
    valid = valid.numpy()
    return interp_depth, valid


def image_path_to_rendered_depth_path(image_name):
    name = image_name.replace("color", "depth")
    return name

def project_to_image(p3D, R, t, camera, eps: float = 1e-4, pad: int = 1):
    p3D = (p3D @ R.T) + t
    visible = p3D[:, -1] >= eps  # keep points in front of the camera
    K = np.array([[camera.params[0], 0, camera.params[2]],
                  [0, camera.params[1], camera.params[3]],
                  [0, 0, 1]])
    p2D_homogeneous = p3D[:, :-1] / p3D[:, -1:].clip(min=eps)
    p2D_homogeneous = np.concatenate([p2D_homogeneous, np.ones((p2D_homogeneous.shape[0], 1))], axis=1)
    p2D = p2D_homogeneous @ K.T
    size = np.array([camera.width - pad - 1, camera.height - pad - 1])
    valid = np.all((p2D[:, :2] >= pad) & (p2D[:, :2] <= size), -1)
    valid &= visible
    return p2D[valid, :2], valid


def correct_sfm_with_depth(sfm_path, depth_folder_path, output_path):
    cameras, images, points3D = read_model(sfm_path)
    for imgid, img in tqdm(images.items()):
        if len(images.items()) < 5: break  # unstable
        image_name = img.name
        depth_name = image_path_to_rendered_depth_path(image_name)
        depth_path = Path(depth_folder_path) / depth_name

        depth = PIL.Image.open(depth_path)
        depth = np.array(depth).astype("float64")
        depth = depth / 1000.0  # mm to meter
        depth[(depth < 0.0001) | (depth > 1000.0)] = np.nan

        R_w2c, t_w2c = img.qvec2rotmat(), img.tvec
        camera = cameras[img.camera_id]
        p3D_ids = img.point3D_ids
        if len(img.point3D_ids[img.point3D_ids != -1]) < 1:
            continue
        p3Ds = np.stack([points3D[i].xyz for i in p3D_ids[p3D_ids != -1]], 0)

        p2Ds, valids_projected = project_to_image(p3Ds, R_w2c, t_w2c, camera)
        invalid_p3D_ids = p3D_ids[p3D_ids != -1][~valids_projected]
        interp_depth, valids_backprojected = interpolate_depth(depth, p2Ds)
        scs = scene_coordinates(
            p2Ds[valids_backprojected],
            R_w2c,
            t_w2c,
            interp_depth[valids_backprojected],
            camera,
        )
        invalid_p3D_ids = np.append(
            invalid_p3D_ids,
            p3D_ids[p3D_ids != -1][valids_projected][~valids_backprojected],
        )
        for p3did in invalid_p3D_ids:
            if p3did == -1:
                continue
            else:
                obs_imgids = points3D[p3did].image_ids
                invalid_imgids = list(np.where(obs_imgids == img.id)[0])
                points3D[p3did] = points3D[p3did]._replace(
                    image_ids=np.delete(obs_imgids, invalid_imgids),
                    point2D_idxs=np.delete(
                        points3D[p3did].point2D_idxs, invalid_imgids
                    ),
                )

        new_p3D_ids = p3D_ids.copy()
        sub_p3D_ids = new_p3D_ids[new_p3D_ids != -1]
        valids = np.ones(np.count_nonzero(new_p3D_ids != -1), dtype=bool)
        valids[~valids_projected] = False
        valids[valids_projected] = valids_backprojected
        sub_p3D_ids[~valids] = -1
        new_p3D_ids[new_p3D_ids != -1] = sub_p3D_ids
        img = img._replace(point3D_ids=new_p3D_ids)

        assert len(img.point3D_ids[img.point3D_ids != -1]) == len(
            scs
        ), f"{len(scs)}, {len(img.point3D_ids[img.point3D_ids != -1])}"
        for i, p3did in enumerate(img.point3D_ids[img.point3D_ids != -1]):
            #points3D[p3did] = points3D[p3did]._replace(xyz=scs[i])
            points3D[p3did] = points3D[p3did]
        images[imgid] = img

    output_path.mkdir(parents=True, exist_ok=True)
    write_model(cameras, images, points3D, output_path)

def create_reference_sfm(full_model, ref_model, whitelist=None, ext=".bin"):
    """Create a new COLMAP model with only training images."""
    logger.info("Creating the reference model.")
    ref_model.mkdir(exist_ok=True)
    cameras, images, points3D = read_model(full_model, ext)

    if whitelist is not None:
        with open(whitelist, "r") as f:
            whitelist = f.read().rstrip().split("\n")

    images_ref = dict()
    for id_, image in images.items():
        if whitelist and image.name not in whitelist:
            continue
        images_ref[id_] = image

    points3D_ref = dict()
    for id_, point3D in points3D.items():
        ref_ids = [i for i in point3D.image_ids if i in images_ref]
        if len(ref_ids) == 0:
            continue
        points3D_ref[id_] = point3D._replace(image_ids=np.array(ref_ids))

    write_model(cameras, images_ref, points3D_ref, ref_model, ".bin")
    logger.info(f"Kept {len(images_ref)} images out of {len(images)}.")
  

def modify_camera_model(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            if line.startswith('#') or line.strip() == "":
                file.write(line)
            else:
                elems = line.split()
                if elems[1] == "SIMPLE_RADIAL":
                    # SIMPLE_RADIAL format: CAMERA_ID SIMPLE_RADIAL WIDTH HEIGHT fx cx cy k1
                    # To convert to PINHOLE, CAMERA_ID PINHOLE WIDTH HEIGHT fx fy cx cy
                    fx = elems[4]
                    fy = elems[4]  
                    cx = elems[5]
                    cy = elems[6]
                    new_line = f"{elems[0]} PINHOLE {elems[2]} {elems[3]} {fx} {fy} {cx} {cy}\n"
                    file.write(new_line)
                else:
                    file.write(line)


def create_query_list_with_intrinsics(
    original, resized, out, list_file=None, ext=".bin", image_dir=None
):
    """Create a list of query images with intrinsics from the colmap model."""
    if ext == ".bin":
        images = read_images_binary(original / "images.bin")
        cameras = read_cameras_binary(resized / "cameras.bin")
    else:
        images = read_images_text(original / "images.txt")
        cameras = read_cameras_text(resized / "cameras.txt")

    name2id = {image.name: i for i, image in images.items()}
    if list_file is None:
        names = list(name2id)
    else:
        with open(list_file, "r") as f:
            names = f.read().rstrip().split("\n")
    data = []
    for name in names:
        image = images[name2id["color_0000.png"]]
        camera = cameras[image.camera_id]
        w, h, params = camera.width, camera.height, camera.params

        if image_dir is not None:
            # Check the original image size and rescale the camera intrinsics
            img = cv2.imread(str(image_dir / name))
            assert img is not None, image_dir / name
            h_orig, w_orig = img.shape[:2]
            assert camera.model == "SIMPLE_RADIAL"
            sx = w_orig / w
            sy = h_orig / h
            assert sx == sy, (sx, sy)
            w, h = w_orig, h_orig
            params = params * np.array([sx, sx, sy, 1.0])

        p = [name, camera.model, w, h] + params.tolist()
        data.append(" ".join(map(str, p)))
    with open(out, "w") as f:
        f.write("\n".join(data))

def run_scene(
    input,
    output,
    images,
    gt_dir,
    results,
    depth_dir,
    num_covis,
    num_retrieve,
):
    output.mkdir(exist_ok=True, parents=True)
    ref_sfm_sift = output / "sfm_sift"
    ref_sfm = output / "sfm_superpoint+superglue"
    query_list = output / "query_list_with_intrinsics.txt"

    feature_conf = {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 8192,
        },
        "preprocessing": {
            "globs": ["*.png"],
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    matcher_conf = match_features.confs["superglue"]
    matcher_conf["model"]["sinkhorn_iterations"] = 5

    train_list = input / "train.txt"
    test_list = input / "test.txt"
    create_reference_sfm(gt_dir, ref_sfm_sift, train_list)

    create_query_list_with_intrinsics(gt_dir, gt_dir, query_list, test_list, ext=".bin")

    features = extract_features.main(feature_conf, images, output, as_half=True)

    sfm_pairs = output / f"pairs-db-covis{num_covis}.txt"
    pairs_from_covisibility.main(ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], output
    )

    if not ref_sfm.exists():
        triangulation.main(
            ref_sfm, ref_sfm_sift, images, sfm_pairs, features, sfm_matches
        )

    ref_sfm_fix = output / "sfm_superpoint+superglue+depth"
    if args.use_dense_depth:
        assert depth_dir is not None
        ref_sfm_fix = output / "sfm_superpoint+superglue+depth"
        correct_sfm_with_depth(ref_sfm, depth_dir, ref_sfm_fix)
        ref_sfm = ref_sfm_fix

    retrieval_conf = extract_features.confs["netvlad"]
    global_descriptors = extract_features.main(retrieval_conf, images, output)

    retrieval_pairs = output / f"pairs-netvlad-retrieve.txt"
    pairs_from_retrieval.main(global_descriptors, retrieval_pairs, num_matched=num_retrieve, query_list=test_list)
    loc_matches = match_features.main(
        matcher_conf, retrieval_pairs, feature_conf["output"], output
    )

    localize_sfm.main(
        ref_sfm,
        query_list,
        retrieval_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=False,
    )


def evaluate(model, results, list_file=None, only_localized=False):
    predictions = {}
    with open(results, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            predictions[name] = (qvec2rotmat(q), t)

    images = read_images_text(model / "images.txt")
    name2id = {image.name: i for i, image in images.items()}

    if list_file is None:
        test_names = list(name2id)
    else:
        with open(list_file, "r") as f:
            test_names = f.read().rstrip().split("\n")

    errors_t = []
    errors_R = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
        else:
            image = images[name2id[name]]
            R_gt, t_gt = image.qvec2rotmat(), image.tvec
            R, t = predictions[name]
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
        errors_t.append(e_t)
        errors_R.append(e_R)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f"Results for file {results.name}:"
    out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"

    out += "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%"
    logger.info(out)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default="E:/room2",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="output/room",
        help="Path to the output directory, default: %(default)s",
    )

    parser.add_argument("--use_dense_depth", 
                        default=False,
                        action="store_true")

    parser.add_argument(
        "--num_covis",
        type=int,
        default=30,
        help="Number of image pairs for SfM, default: %(default)s",
    )

    parser.add_argument(
        "--num_retrieve",
        type=int,
        default=5,
        help="Number of images for retrieval, default: %(default)s",
    )
    args = parser.parse_args()

    gt_dirs = args.dataset / "views/sparse/0" 

    all_results = {}

    logger.info(f'Working on scene.')

    results = (
    args.outputs 
    / "results_{}.txt".format("dense" if args.use_dense_depth else "sparse")
    )

    if args.overwrite or True:
        run_scene(
            args.dataset ,
            args.outputs ,
            args.dataset / "images",
            Path(gt_dirs),
            results,
            args.dataset / "depths",
            args.num_covis,
            args.num_retrieve
            )
    all_results = results
            
    for scene in args.scenes:
        logger.info(f'Evaluate scene "{scene}".')    
        gt_dir = Path(str(gt_dirs).format(scene=scene))
        list_file = args.dataset / "test.txt"
        evaluate(gt_dir, all_results[scene], list_file)
