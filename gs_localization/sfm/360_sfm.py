import argparse
import sys
from pathlib import Path
import logging
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
from hloc.utils.read_write_model import read_model, write_model, read_images_text, qvec2rotmat

logger = logging.getLogger(__name__)
    
def run_scene(
    images,
    gt_dir,
    tr_dir,
    input,
    output,
    results,
    num_covis,
    num_retrieve
):
    output.mkdir(exist_ok=True, parents=True)
    ref_sfm_sift =  input / "train_views/triangulated"
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
            "globs": ["*.JPG"],
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    matcher_conf = match_features.confs["superglue"]
    matcher_conf["model"]["sinkhorn_iterations"] = 5

    test_list = tr_dir / "list_test.txt"
    create_query_list_with_intrinsics(gt_dir, tr_dir, query_list, test_list)

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

def evaluate(scene_dir, model, results, list_file=None, only_localized=False):
    poses_bounds = np.load(scene_dir/'poses_bounds.npy') # (N_images, 17)

    bounds = poses_bounds[:, -2:]  # (N_images, 2)

    # correct scale, See https://github.com/bmild/nerf/issues/34
    near_original = bounds.min()
    scale_factor = near_original * 0.75  # 0.75 is the default parameter

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
        errors_t.append(e_t/scale_factor)
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
        ratio_t = np.mean((errors_t < th_t))
        ratio_R = np.mean((errors_R < th_R))
        out += f"\n\t{th_t}unit, {th_R}deg : {ratio_t*100:.2f}, {ratio_R*100:.2f}%"
    logger.info(out)
    print(out)


SCENES = ['bicycle', 'bonsai', 'counter', 'garden',  'kitchen', 'room', 'stump', "flowers", "treehill"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=SCENES, choices=SCENES, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/360_v2",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="output/360_v2",
        help="Path to the output directory, default: %(default)s",
    )

    parser.add_argument(
        "--num_covis",
        type=int,
        default=30,
        help="Number of image pairs for SfM, default: %(default)s",
    )

    parser.add_argument(
        "--num_retrieve",
        type=int,
        default=13,
        help="Number of images for retrieval, default: %(default)s",
    )
    args = parser.parse_args()

    gt_dirs = args.dataset / "{scene}/sparse/0" 
    tr_dirs = args.dataset / "{scene}/train_views/triangulated" 

    all_results = {}
    for scene in args.scenes:
        logger.info(f'Working on scene "{scene}".')

        results = (
        args.outputs / scene
        / "results_{}.txt".format("dense" if False else "sparse")
        )

        if args.overwrite or True:
            run_scene(
                args.dataset / scene / "images_4",
                Path(str(gt_dirs).format(scene=scene)),
                Path(str(tr_dirs).format(scene=scene)), 
                args.dataset / scene,
                args.outputs / scene,
                results,
                args.num_covis,
                args.num_retrieve)

        all_results[scene] = results
            
    for scene in args.scenes:
        logger.info(f'Evaluate scene "{scene}".')    
        gt_dir = Path(str(gt_dirs).format(scene=scene))
        list_file = args.dataset / f"{scene}/train_views/triangulated/list_test.txt" 
        evaluate(args.dataset / f"{scene}", gt_dir, all_results[scene], list_file)
