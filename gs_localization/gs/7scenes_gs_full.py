import torch
import shutil
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from random import randint, shuffle
import uuid
from tqdm import tqdm

import os
import gc
import sys
from pathlib import Path
gaussian_splatting_path = Path(__file__).resolve().parent.parent.parent / 'gaussian_splatting'
sys.path.append(str(gaussian_splatting_path))

from scene.scene_batch import Scene, GaussianModel
from gaussian_renderer import render, network_gui
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.depth_utils import estimate_depth
from torchmetrics.functional.regression import pearson_corrcoef

def create_symlink(source_path, target_path):
    if os.path.exists(target_path):
        os.remove(target_path)  
    os.symlink(source_path, target_path) 

def copy_images_and_setup_sfm(dataset, outputs, scene):
    # Define the source directory for images and the destination directory
    images_src_dir = Path(dataset) / scene / 'train_images_full'
    images_dst_dir = Path(outputs) / scene / 'images'
    depths_src_dir = Path(dataset) / scene / 'train_depths_full'
    depths_dst_dir = Path(outputs) / scene / 'depths'

    sfm_src_dir = Path(outputs) / scene / 'sfm_superpoint+superglue'
    sparse_dst_dir = Path(outputs) / scene / 'sparse/0'

    # Create the destination images directory if it doesn't exist
    images_dst_dir.mkdir(parents=True, exist_ok=True)
    depths_dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy all image files from the source directory to the destination directory
    for image_file in images_src_dir.glob('*'):
        if image_file.is_file():
            create_symlink(image_file, images_dst_dir / image_file.name)

    for depth_file in depths_src_dir.glob('*'):
        if depth_file.is_file():
            create_symlink(depth_file, depths_dst_dir / depth_file.name)

    # Create sparse/0 directory
    sparse_dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from the sfm_superpoint+superglue directory to the sparse/0 directory
    for sfm_file in sfm_src_dir.glob('*'):
        if sfm_file.is_file():
            create_symlink(sfm_file, sparse_dst_dir / sfm_file.name)

def delete_directories(outputs, scene):
    # Define the paths for the images, depths, and sparse/0 directories
    images_dir = Path(outputs) / scene / 'images'
    depths_dir = Path(outputs) / scene / 'depths'
    sparse_dir = Path(outputs) / scene / 'sparse'

    # Delete the images directory if it exists
    if images_dir.exists() and images_dir.is_dir():
        shutil.rmtree(images_dir)
        print(f"Deleted directory: {images_dir}")
    else:
        print(f"Directory does not exist: {images_dir}")

    # Delete the depths directory if it exists
    if depths_dir.exists() and depths_dir.is_dir():
        shutil.rmtree(depths_dir)
        print(f"Deleted directory: {depths_dir}")
    else:
        print(f"Directory does not exist: {depths_dir}")

    # Delete the sparse/0 directory if it exists
    if sparse_dir.exists() and sparse_dir.is_dir():
        shutil.rmtree(sparse_dir)
        print(f"Deleted directory: {sparse_dir}")
    else:
        print(f"Directory does not exist: {sparse_dir}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack, pseudo_stack, generate_pseudo_stack, counter = None, None, True, 0
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()
            viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack)-1)]
            if viewpoint_cam.depth != None:
                use_depth = True
            else:
                use_depth = False
        viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack)-1)]

        counter += 1
        if counter == dataset.batch*4-1:
            del viewpoint_stack
            torch.cuda.empty_cache()
            viewpoint_stack = None
            counter = 0

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # pseudo_depth loss
        depth = render_pkg["depth"][0]
        midas_depth = viewpoint_cam.pseudo_depth.clone().detach()
        depth = depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)
        pseudo_depth_loss = min(
                        (1 - pearson_corrcoef(-midas_depth, depth)),
                        (1 - pearson_corrcoef(1000 / (midas_depth + 200.), depth))
        )
        loss += 0.01 * pseudo_depth_loss

        # l1 depth loss
        if use_depth == True:
            gt_depth = viewpoint_cam.depth.clone().detach().reshape(-1,1)
            depth_mask = (gt_depth>0)
            #loss += (-1e-6 * iteration + 0.05) * l1_loss(depth*depth_mask, gt_depth*depth_mask)
            loss += 0.05 * l1_loss(depth*depth_mask, gt_depth*depth_mask)
            
        # pseudo_view regularization
        if generate_pseudo_stack and (not pseudo_stack):
            pseudo_stack = scene.getPseudoCameras().copy()
            if pseudo_stack == [None]:
                generate_pseudo_stack = False
        
        if generate_pseudo_stack and pseudo_stack and iteration % args.sample_pseudo_interval == 0 \
                                and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, bg)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"]).detach()

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = min(
                        (1 - pearson_corrcoef(-midas_depth_pseudo, rendered_depth_pseudo)),
                        (1 - pearson_corrcoef(1000 / (midas_depth_pseudo + 200.), rendered_depth_pseudo))
                        )
            loss += 0.005 * depth_loss_pseudo

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, bg), viewpoint_stack)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")



def training_too_large(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, indices):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, indices=indices[-3000:])
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack, pseudo_stack, generate_pseudo_stack, counter = None, None, True, 0
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1): 
        if iteration == 11900:
            del scene
            del viewpoint_stack
            
            scene = Scene(dataset, gaussians, load_iteration="continue", indices=indices[:4000])
            viewpoint_stack, pseudo_stack, generate_pseudo_stack, counter = None, None, True, 0

        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()
            viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack)-1)]
            if viewpoint_cam.depth != None:
                use_depth = True
            else:
                use_depth = False
        viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack)-1)]

        counter += 1
        if counter == dataset.batch*4-1:
            del viewpoint_stack
            torch.cuda.empty_cache()
            viewpoint_stack = None
            counter = 0

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # pseudo_depth loss
        depth = render_pkg["depth"][0]
        midas_depth = viewpoint_cam.pseudo_depth.clone().detach()
        depth = depth.reshape(-1, 1)
        midas_depth = midas_depth.reshape(-1, 1)
        pseudo_depth_loss = min(
                        (1 - pearson_corrcoef(-midas_depth, depth)),
                        (1 - pearson_corrcoef(1000 / (midas_depth + 200.), depth))
        )
        loss += 0.01 * pseudo_depth_loss

        # l1 depth loss
        if use_depth == True:
            gt_depth = viewpoint_cam.depth.clone().detach().reshape(-1,1)
            depth_mask = (gt_depth>0)
            #loss += (-1e-6 * iteration + 0.05) * l1_loss(depth*depth_mask, gt_depth*depth_mask)
            loss += 0.05 * l1_loss(depth*depth_mask, gt_depth*depth_mask)
            
        # pseudo_view regularization
        if generate_pseudo_stack and (not pseudo_stack):
            pseudo_stack = scene.getPseudoCameras().copy()
            if pseudo_stack == [None]:
                generate_pseudo_stack = False
        
        if generate_pseudo_stack and pseudo_stack and iteration % args.sample_pseudo_interval == 0 \
                                and iteration > args.start_sample_pseudo and iteration < args.end_sample_pseudo:
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))

            render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, bg)
            rendered_depth_pseudo = render_pkg_pseudo["depth"][0]
            midas_depth_pseudo = estimate_depth(render_pkg_pseudo["render"]).detach()

            rendered_depth_pseudo = rendered_depth_pseudo.reshape(-1, 1)
            midas_depth_pseudo = midas_depth_pseudo.reshape(-1, 1)
            depth_loss_pseudo = min(
                        (1 - pearson_corrcoef(-midas_depth_pseudo, rendered_depth_pseudo)),
                        (1 - pearson_corrcoef(1000 / (midas_depth_pseudo + 200.), rendered_depth_pseudo))
                        )
            loss += 0.005 * depth_loss_pseudo

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, bg), viewpoint_stack)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, viewpoint_stack):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras': viewpoint_stack},]
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                depth_test = 0.0
                pearson_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    depth = renderFunc(viewpoint, scene.gaussians, *renderArgs)["depth"][0].reshape(-1,1)
                    gt_depth = viewpoint.depth.clone().detach().reshape(-1,1)
                    midas_depth = viewpoint.pseudo_depth.clone().detach().reshape(-1,1)
                    depth_mask = (gt_depth>0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                            image[None], 
                                            global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                            gt_image[None], 
                                            global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    depth_test += l1_loss(depth*depth_mask, gt_depth*depth_mask)
                    pearson_test += min(
                        (1 - pearson_corrcoef(-midas_depth, depth)),
                        (1 - pearson_corrcoef(1000 / (midas_depth + 200.), depth))
                        )
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} Depth {} Pearson {}".format(
                            iteration, config['name'], l1_test, psnr_test, 0.02*depth_test, 0.01*pearson_test))
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
SCENES = ["office", "redkitchen"]
SCENES = ["chess", "fire", "heads", "pumpkin", "stairs"]

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument(
        "--dataset",
        type=Path,
        default="D:/gs-localization/datasets/7scenes",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="D:/gs-localization/output/7scenes_full",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    network_gui.init(args.ip, args.port)
    for scene in SCENES:
        copy_images_and_setup_sfm(args.dataset, args.outputs, scene)
        print("Optimizing ", args.model_path)

        # Update lp with the correct paths for the current scene
        args.source_path = Path(args.outputs)/scene
        args.model_path = Path(args.outputs)/scene

        # Initialize system state (RNG)
        safe_state(args.quiet)

        # set sh degree 
        # args.sh_degree = 3

        # Configure and run training
        torch.autograd.set_detect_anomaly(args.detect_anomaly)

    
        if scene in ["office"]:
            numbers = list(range(6000))
            shuffle(numbers)

            training_too_large(lp.extract(args), 
                op.extract(args), 
                pp.extract(args), 
                [3_000, 7_000, 10_000, 15_000, 20_000, 25_000, 27_500, 30_000], 
                [7_000, 30_000], 
                args.checkpoint_iterations, 
                args.start_checkpoint, 
                args.debug_from,
                indices=numbers)
            
        elif scene in ["redkitchen"]:
            numbers = list(range(7000))
            shuffle(numbers)

            training_too_large(lp.extract(args), 
                op.extract(args), 
                pp.extract(args), 
                [3_000, 7_000, 10_000, 15_000, 20_000, 25_000, 27_500, 30_000], 
                [7_000, 30_000], 
                args.checkpoint_iterations, 
                args.start_checkpoint, 
                args.debug_from,
                indices=numbers)


        else:
            training(lp.extract(args), 
                    op.extract(args), 
                    pp.extract(args), 
                    [3_000, 7_000, 10_000, 15_000, 20_000, 25_000, 27_500, 30_000], 
                    [7_000, 30_000], 
                    args.checkpoint_iterations, 
                    args.start_checkpoint, 
                    args.debug_from
                    )
        
        # All done
        print("\nTraining complete.")

        delete_directories(args.outputs, scene)


        # Define the path for the point_cloud directory
        point_cloud_dir = Path(args.outputs) / scene / 'point_cloud'
        gs_map_dir = Path(args.outputs) / scene / 'gs_map'

        # Rename the point_cloud directory to gs_map
        if point_cloud_dir.exists() and point_cloud_dir.is_dir():
            if gs_map_dir.exists() and gs_map_dir.is_dir():
                shutil.rmtree(gs_map_dir)
            point_cloud_dir.rename(gs_map_dir)
            print(f"Renamed directory: {point_cloud_dir} to {gs_map_dir}")
        else:
            print(f"Directory does not exist: {point_cloud_dir}")
