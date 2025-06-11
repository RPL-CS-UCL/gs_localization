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

logger = logging.getLogger(__name__)
    
def run_scene(
    images,
    gt_dir,
    tr_dir,
    input,
    output,
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
            "max_keypoints": 4096,
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


    results = (
        output
        / "results_{}.txt".format("dense" if False else "sparse")
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


SCENES = ['bicycle', 'bonsai', 'counter', 'garden',  'kitchen', 'room', 'stump']


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
        default=1,
        help="Number of images for retrieval, default: %(default)s",
    )
    args = parser.parse_args()

    gt_dirs = args.dataset / "{scene}/sparse/0" 
    tr_dirs = args.dataset / "{scene}/train_views/triangulated" 

    for scene in args.scenes:
        logger.info(f'Working on scene "{scene}".')
        if args.overwrite or True:
            run_scene(
                args.dataset / scene / "images_4",
                Path(str(gt_dirs).format(scene=scene)),
                Path(str(tr_dirs).format(scene=scene)), 
                args.dataset / scene,
                args.outputs / scene,
                args.num_covis,
                args.num_retrieve)

