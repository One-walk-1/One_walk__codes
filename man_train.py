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
import datetime
import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.generate_camera import generate_new_cam, quat_xyzw_to_rotmat_torch, generate_new_cam_torch

import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
from utils.logger import logger_config 
from scipy.spatial.transform import Rotation
from utils.data_painter import paint_spectrum_compare 
from skimage.metrics import structural_similarity as ssim

import collections




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    datadir = 'data'
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join('logs', current_time)
    log_filename = "logger.log"
    devices = torch.device('cuda')
    log_savepath = os.path.join(logdir, log_filename)
    os.makedirs(logdir,exist_ok=True)
    logger = logger_config(log_savepath=log_savepath, logging_name='gsss')
    logger.info("datadir:%s, logdir:%s", datadir, logdir)
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,current_time)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.gaussian_init(random = False)
    deform = DeformModel(train_man = False)
    deform.train_setting(opt)
    
    scene = Scene(dataset, gaussians)
    scene.dataset_init()
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        deform_ckpt_path = checkpoint.replace("chkpnt", "deform_chkpnt")
        deform_ckpt = torch.load(deform_ckpt_path)
        deform.deform.load_state_dict(deform_ckpt["deform_params"])
        print("Loaded checkpoint from {}, starting at iteration {}".format(checkpoint, first_iter))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = None
    max_grad_norm = 0.5
    accumulation_steps = 4
    accumulation_counter = 0
                
    gaussians_man = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians_man.gaussian_init(random = True)
    deform_man = DeformModel(train_man = True)
    deform_man.train_setting(opt)
    gaussians_man.training_setup(opt)
    
    man_iter_start = torch.cuda.Event(enable_timing = True)
    man_iter_end = torch.cuda.Event(enable_timing = True)
    
    ema_loss_for_log_man = 0.0
    first_iter = 1
    progress_bar_man = tqdm(range(first_iter, opt.iterations + 1), desc="Training progress man")
    SSIM_queue_man = collections.deque(maxlen=600)
    accumulation_counter = 0
    for iteration in range(first_iter, opt.iterations + 1):
        train_step(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
                   devices, background, man_iter_start, man_iter_end, ema_loss_for_log_man, progress_bar_man, SSIM_queue_man,
                   iteration, scene, gaussians_man, deform_man, max_grad_norm, accumulation_steps, accumulation_counter,
                   train_man=True, gaussians_bg=gaussians, deform_bg=deform)
        
        accumulation_counter += 1
        
        if iteration in testing_iterations:
            with torch.no_grad():
                test(scene, gaussians_man, deform_man, pipe, background, logger, logdir, iteration, devices, 
                     train_man=True, gaussians_bg=gaussians, deform_bg=deform, val_man=True)
                test(scene, gaussians_man, deform_man, pipe, background, logger, logdir, iteration, devices, 
                     train_man=True, gaussians_bg=gaussians, deform_bg=deform, val_man=False)
                
                
        # # test 
        # if iteration in testing_iterations:
        #     with torch.no_grad():
        #         test(scene, gaussians, deform, pipe, bg, logger, logdir, 'testset', iteration)
        #         test(scene, gaussians, deform, pipe, bg, logger, logdir, 'trainset', iteration)
    progress_bar_man.close()    
                
def train_step(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
               devices, background, iter_start, iter_end, ema_loss_for_log, progress_bar, SSIM_queue,
               iteration, scene, gaussians, deform, max_grad_norm, accumulation_steps, accumulation_counter,
               train_man = False, gaussians_bg = None, deform_bg = None
               ):
    iter_start.record()

    gaussians.update_learning_rate(iteration)

    # Every 1000 its we increase the levels of SH up to a maximum degree
    if iteration % 1000 == 0:
        gaussians.oneupSHdegree()
    if iteration % 1000 == 0:
        print("nums of gaussians:", gaussians.get_xyz.shape[0])

    # Pick a random Camera
    if not train_man:
        try:
            # spectrum, tx_pos = next(scene.train_iter_dataset)
            spectrum, rx_pos, rx_orientation = next(scene.train_iter_dataset)
        except:
            scene.dataset_init()
            # spectrum, tx_pos = next(scene.train_iter_dataset)
            spectrum, rx_pos, rx_orientation = next(scene.train_iter_dataset)
    else: 
        try:
            spectrum, rx_pos, man_pos, man_orientation = next(scene.train_iter_man_dataset)
        except:
            scene.dataset_init()
            spectrum, rx_pos, man_pos, man_orientation = next(scene.train_iter_man_dataset)

    if not train_man:
        rx_orientation = rx_orientation.squeeze(0) 
        R = torch.from_numpy(Rotation.from_quat(rx_orientation).as_matrix()).float()
        # print(R.shape)
        rx_pos = rx_pos.cuda()
        rx_pos_cpu = rx_pos.cpu().numpy()  
        viewpoint_cam = generate_new_cam(R, rx_pos_cpu)
        N = gaussians.get_xyz.shape[0]
        rx_input = rx_pos.expand(N, -1) 
        rx_orientation_cuda = rx_orientation.cuda()
        rx_orient_input = rx_orientation_cuda.expand(N, -1)
        
        d_signal = deform.step(gaussians.get_xyz.detach(), rx_input, rx_orient_input).squeeze(-1) 
        d_xyz, d_rotation, d_scaling = 0, 0, 0
    
    else:
        rx_orientation = [0, 0, 0, 1.0]
        R = torch.from_numpy(Rotation.from_quat(rx_orientation).as_matrix()).float()
        rx_orientation = torch.tensor(rx_orientation)
        
        rx_pos = rx_pos.cuda()
        rx_pos_cpu = rx_pos.cpu().numpy()  
        viewpoint_cam = generate_new_cam(R, rx_pos_cpu)
        
        N = gaussians.get_xyz.shape[0]
        rx_input = rx_pos.expand(N, -1) 
        man_pos = man_pos.cuda()
        man_orientation = man_orientation.cuda()
        man_pos = man_pos.expand(N, -1)
        man_orientation = man_orientation.expand(N, -1)

        # d_xyz, d_scaling, d_rotation, d_att, d_signal = deform.step_man(gaussians.get_xyz.detach(), rx_input, man_pos, man_orientation)
        d_xyz, d_scaling, d_rotation, d_signal = deform.step_man(gaussians.get_xyz.detach(), rx_input, man_pos, man_orientation)
        d_signal = d_signal.squeeze(-1)

    if train_man:
        N_bg = gaussians_bg.get_xyz.shape[0]
        
    # Render
    if (iteration - 1) == debug_from:
        pipe.debug = True

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    if not train_man:
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,d_xyz, d_rotation, d_scaling, d_signal, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
    else:
        rx_input_bg = rx_pos.expand(N_bg, -1) 
        rx_orientation_cuda = rx_orientation.cuda()
        # rx_orient_input_bg = rx_orientation_cuda.expand(N_bg, -1) 
        d_signal_bg= deform_bg.step(gaussians_bg.get_xyz.detach(), rx_input_bg).squeeze(-1)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,d_xyz, d_rotation, d_scaling, d_signal, gaussians_bg, d_signal_bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        
    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        
    channel,height, width = image.shape
    image_masked = image[0,:height, :]
    render_image_show = image_masked.reshape( 1, 90, 360).cuda()
    pred_spectrum_real = image[0,:height, :]
    pred_spectrum_imag = image[1,:height, :]
    pred_spectrum = pred_spectrum_real + 1j * pred_spectrum_imag
    pred_spectrum = torch.abs(pred_spectrum)
    image=pred_spectrum

    current_lambda = opt.lambda_dssim
    if iteration < opt.iterations * 0.3:
        current_lambda = 0.2  
    elif iteration < opt.iterations * 0.7:
        current_lambda = 0.4  
    else:
        current_lambda = 0.6  

    def max_distance_loss(points, max_dist_threshold=0.15):
        dist_matrix = torch.cdist(points, points)
        max_dist = torch.max(dist_matrix, dim=1)[0]  
        max_dist_penalty = torch.sum(torch.relu(max_dist - max_dist_threshold))  
        return max_dist_penalty

    # Loss
    gt_image = spectrum.cuda()
    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(image.unsqueeze(0).unsqueeze(0), gt_image.unsqueeze(0).unsqueeze(0))
    else:
        ssim_value = ssim(image, gt_image)

    loss = (1.0 - current_lambda) * Ll1 + current_lambda * (1.0 - ssim_value)
    
    Ll1depth_pure = 0.0
        
    Ll1depth = 0

    loss.backward()

    iter_end.record()
    
    with torch.no_grad():
        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        # ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

        SSIM_queue.append(ssim_value.cpu().numpy())
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", 
                                    # "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                                    #   "Mean pixel error": f"{pixel_error:.6f}",
                                    "SSIM": f"{ssim_value:.6f}", "Median SSIM": f"{np.median(SSIM_queue):.6f}", "Mean SSIM": f"{np.mean(SSIM_queue):.6f}"
                                    })
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

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
                gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
            if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                gaussians.reset_opacity()

        # Optimizer step
        if iteration < opt.iterations:
            accumulation_counter += 1

            if accumulation_counter % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(deform.parameters(), max_grad_norm)

                gaussians.exposure_optimizer.step()
                gaussians.optimizer.step()
                deform.optimizer.step()

                gaussians.update_learning_rate(iteration)
                deform.update_learning_rate(iteration)

                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()

                accumulation_counter = 0
                
        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
def test(scene, gaussians, deform, pipe, bg, logger, logdir, iteration, devices=torch.device('cuda'), 
         train_man=False, gaussians_bg=None, deform_bg=None, val_man=False):
    torch.cuda.empty_cache()
    if not train_man:            
        logger.info("Start evaluation")
    else:
        if not val_man:
            logger.info("Start test man")
        else:
            logger.info("Start validation man")
    
    if not train_man:  
        iteration_path = os.path.join(logdir, 'pred_spectrum', str(iteration))
        os.makedirs(iteration_path, exist_ok=True) 
        full_path = os.path.join(logdir, str(iteration))
        os.makedirs(full_path, exist_ok=True)
    else:
        if not val_man:
            iteration_path = os.path.join(logdir, 'pred_spectrum_with_man_test', str(iteration))
            os.makedirs(iteration_path, exist_ok=True) 
            full_path = os.path.join(logdir, str(iteration))
            os.makedirs(full_path, exist_ok=True)
        else:
            iteration_path = os.path.join(logdir, 'pred_spectrum_with_man_val', str(iteration))
            os.makedirs(iteration_path, exist_ok=True) 
            full_path = os.path.join(logdir, str(iteration))
            os.makedirs(full_path, exist_ok=True)
        
    save_img_idx = 0
    all_ssim = []

    if not train_man:
        for test_input, test_label, test_orientation in scene.test_iter: 
            rx_orientation1 = test_orientation.squeeze(0)  
            R = torch.from_numpy(Rotation.from_quat(rx_orientation1).as_matrix()).float()
            rx_pos_1 = test_label.cuda()
            rx_pos_cpu_1 = rx_pos_1.cpu().numpy()  
            viewpoint_cam = generate_new_cam(R, rx_pos_cpu_1)
            N = gaussians.get_xyz.shape[0]
            rx_input_1 = rx_pos_1.expand(N, -1)
            rx_orientation1_cuda = rx_orientation1.cuda()
            rx_orient_input1 = rx_orientation1_cuda.expand(N, -1)

            d_signal = deform.step(gaussians.get_xyz.detach(), rx_input_1, rx_orient_input1).squeeze(-1)
            d_xyz, d_rotation, d_scaling = 0, 0, 0
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg,d_xyz, d_rotation, d_scaling, d_signal)

            image = render_pkg["render"]
            channel,height, width = image.shape
            pred_spectrum_real = image[0,:height, :]
            pred_spectrum_imag = image[1,:height, :]
            pred_spectrum = pred_spectrum_real + 1j * pred_spectrum_imag
            pred_spectrum = torch.abs(pred_spectrum)

            ## save predicted spectrum
            pred_spectrum = pred_spectrum.detach().cpu().numpy()
            gt_spectrum = test_input.squeeze(0).detach().cpu().numpy()
    
                        
            pixel_error = np.mean(abs(pred_spectrum - gt_spectrum))
            ssim_i = ssim(pred_spectrum, gt_spectrum, data_range=1, multichannel=False)
            logger.info(
                "Spectrum {:d}, Mean pixel error = {:.6f}; SSIM = {:.6f}".format(save_img_idx, pixel_error,
                                                                                            ssim_i))
            paint_spectrum_compare(pred_spectrum, gt_spectrum,
                                save_path=os.path.join(iteration_path,
                                                f'{save_img_idx}.png'))
            all_ssim.append(ssim_i)
            logger.info("Median SSIM is {:.6f}".format(np.median(all_ssim)))
            save_img_idx += 1
            np.savetxt(os.path.join(full_path, 'all_ssim.txt'), all_ssim, fmt='%.4f')
            
    else:
        if not val_man:
            loader = scene.test_iter_man
        else:
            loader = scene.val_iter_man
        for spectrum, rx_pos, man_pos, man_orientation in loader:
            rx_orientation = [0, 0, 0, 1.0]
            R = torch.from_numpy(Rotation.from_quat(rx_orientation).as_matrix()).float()
            rx_orientation = torch.tensor(rx_orientation)
            rx_pos = rx_pos.cuda()
            rx_pos_cpu = rx_pos.cpu().numpy()  
            viewpoint_cam = generate_new_cam(R, rx_pos_cpu)
            
            N = gaussians.get_xyz.shape[0]
            rx_input = rx_pos.expand(N, -1) 
            man_pos = man_pos.cuda()
            man_orientation = man_orientation.cuda()
            man_pos = man_pos.expand(N, -1)
            man_orientation = man_orientation.expand(N, -1)

            # d_xyz, d_scaling, d_rotation, d_att, d_signal = deform.step_man(gaussians.get_xyz.detach(), rx_input, man_pos, man_orientation)
            d_xyz, d_scaling, d_rotation, d_signal = deform.step_man(gaussians.get_xyz.detach(), rx_input, man_pos, man_orientation)
            d_signal = d_signal.squeeze(-1)  
            
            N_bg = gaussians_bg.get_xyz.shape[0]
            
            rx_input_bg = rx_pos.expand(N_bg, -1) 
            # rx_orientation_cuda = rx_orientation.cuda()
            # rx_orient_input_bg = rx_orientation_cuda.expand(N_bg, -1) 
            d_signal_bg= deform_bg.step(gaussians_bg.get_xyz.detach(), rx_input_bg).squeeze(-1)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg,d_xyz, d_rotation, d_scaling, d_signal, gaussians_bg, d_signal_bg)
            
            image = render_pkg["render"]
            channel,height, width = image.shape
            pred_spectrum_real = image[0,:height, :]
            pred_spectrum_imag = image[1,:height, :]
            pred_spectrum = pred_spectrum_real + 1j * pred_spectrum_imag
            pred_spectrum = torch.abs(pred_spectrum)

            ## save predicted spectrum
            pred_spectrum = pred_spectrum.detach().cpu().numpy()
            gt_spectrum = spectrum.squeeze(0).detach().cpu().numpy()
    
                        
            pixel_error = np.mean(abs(pred_spectrum - gt_spectrum))
            ssim_i = ssim(pred_spectrum, gt_spectrum, data_range=1, multichannel=False)
            logger.info(
                "Spectrum {:d}, Mean pixel error = {:.6f}; SSIM = {:.6f}".format(save_img_idx, pixel_error,
                                                                                            ssim_i))
            paint_spectrum_compare(pred_spectrum, gt_spectrum,
                                save_path=os.path.join(iteration_path,
                                                f'{save_img_idx}.png'))
            all_ssim.append(ssim_i)
            logger.info("Median SSIM is {:.6f}".format(np.median(all_ssim)))
            logger.info("Mean SSIM is {:.6f}".format(np.mean(all_ssim)))
            save_img_idx += 1
            if not val_man:
                np.savetxt(os.path.join(full_path, 'all_ssim_test.txt'), all_ssim, fmt='%.4f')
            else:
                np.savetxt(os.path.join(full_path, 'all_ssim_val.txt'), all_ssim, fmt='%.4f')


def prepare_output_and_logger(args,time):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", time)
        
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.74")
    parser.add_argument('--port', type=int, default=6074)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[60000,100000,200000,300000,400000,500000,600000, 800000,1000000,1200000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000,60000,200000,300000,600000,1200000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000, 30000, 60000, 200000, 300000, 600000,1200000])
    parser.add_argument("--start_checkpoint", type=str, default = "checkpoints/chkpnt200000.pth")
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args(sys.argv[1:])
    
    args.save_iterations.append(args.iterations)
    torch.cuda.set_device(args.gpu)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
