from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import datasets.PotholeDataset
import datasets.PotholeDataset2
import datasets.mono_dataset
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))
        
        #filenames = readlines(os.path.join(os.path.dirname(__file__),"test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        fpath = os.path.join(os.path.dirname(__file__), "splits", "C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New\\splits\\pothole\\", "{}_files.txt")
        print(fpath)
        filenames = readlines(fpath.format("test"))



        dataset = datasets.mono_dataset.PotholeDataset3(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                            is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc,opt.scales)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

        decoder_state_dict = torch.load(decoder_path)
        print(decoder_state_dict.keys())
        print(depth_decoder.state_dict().keys())


        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    #gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    #gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    # Extract ground truth depth maps from the dataset
    ground_truth_depths = []
    for data in dataloader:
        depth_gt = data[("depth", 0, 0)]  # Assuming your dataset has depth here
        depth_gt = depth_gt.cpu().numpy().astype(np.float32)  # Move to CPU and convert to NumPy

        if depth_gt.ndim == 4:  # (Batch, 1, H, W)
            depth_gt = depth_gt[:, 0]  # Remove channel dimension

        ground_truth_depths.extend(depth_gt)  # Collect all depth maps

    gt_depths = np.array(ground_truth_depths)  # Convert to NumPy array


    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

# def evaluate(opt):
#     """Evaluates a pretrained model using a specified test set"""
#     MIN_DEPTH = 0.1  # Adjust based on training
#     MAX_DEPTH = 100  # Adjust based on dataset

#     assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
#         "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

#     opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
#     assert os.path.isdir(opt.load_weights_folder), \
#         f"Cannot find a folder at {opt.load_weights_folder}"

#     print(f"-> Loading weights from {opt.load_weights_folder}")
    
#     # Load test filenames
#     #filenames = readlines(os.path.join(os.path.dirname(__file__), "test_files.txt"))
    
#     # Load encoder and decoder
#     encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
#     decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

#     encoder_dict = torch.load(encoder_path)
#     encoder = networks.ResnetEncoder(opt.num_layers, pretrained=False)
#     encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}, strict=False)
#     #encoder.to(opt.device).eval()
#     encoder.cuda()
#     encoder.eval()

#     depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales)
#     depth_decoder.load_state_dict(torch.load(decoder_path), strict=False)
#     #depth_decoder.to(opt.device).eval()
#     depth_decoder.cuda()
#     depth_decoder.eval()
    

#     fpath = os.path.join(os.path.dirname(__file__), "splits", "C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset\\splits", "{}_files.txt")
#     print(fpath)
#     #train_filenames = readlines(fpath.format("train"))
#     filenames = readlines(fpath.format("test"))

#     # Dataset and Dataloader
#     dataset = datasets.PotholeDataset2.PotholeDataset2(opt.data_path, filenames,
#                                                      encoder_dict['height'], encoder_dict['width'],
#                                                      is_train=False)
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=opt.num_workers,
#                             pin_memory=True, drop_last=False)

#     pred_disps = []
#     ground_truth=[]

#     # Perform inference
#     print(f"-> Computing predictions with size {encoder_dict['width']}x{encoder_dict['height']}")
#     with torch.no_grad():
#         for data in dataloader:
#             input_color = data[("color", 0, 0)].cuda()
#             #depth_color= data[("depth", 0, 0)].cuda()
#             depth_maps = data[("depth", 0, 0)]  # Extract depth maps
#             print("SHape:", depth_maps.shape)
#             depth_maps = depth_maps.cpu().numpy().astype(np.float32)  # Convert to NumPy
#             ground_truth.append(depth_maps)

#             if opt.post_process:
#                 input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

#             output = depth_decoder(encoder(input_color))
#             pred_disp, _ = disp_to_depth(output[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
#             pred_disp = pred_disp.cpu()[:, 0].numpy()

#             if opt.post_process:
#                 N = pred_disp.shape[0] // 2
#                 pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

#             pred_disps.append(pred_disp)
#             print(pred_disp.shape)

#     pred_disps = np.concatenate(pred_disps)
#     # if isinstance(ground_truth, list):
#     #     ground_truth = [d.cpu().numpy() if isinstance(d, torch.Tensor) else np.array(d) for d in ground_truth]
#     max_batch_size = max(d.shape[0] for d in ground_truth)  # Find the largest batch size

#     ground_truths = []
#     for d in ground_truth:  # Assuming `original_ground_truth` is the list of depth tensors
#         if isinstance(d, torch.Tensor):
#             d = d.cpu().numpy().astype(np.float32)  # Convert to NumPy

#         if d.ndim == 3:  # If shape is (B, H, W), select the first sample
#             d = d[0]  # Take only the first image from the batch

#         ground_truths.append(d)  # Store corrected depth map

#     # Stack as NumPy array if all shapes match
#     ground_truths = np.array(ground_truths)


#     # Save predictions if needed
#     if opt.save_pred_disps:
#         output_path = os.path.join(opt.load_weights_folder, f"disps_{opt.eval_split}_split.npy")
#         print(f"-> Saving predicted disparities to {output_path}")
#         np.save(output_path, pred_disps)

#     # Evaluate against ground truth
#     print(f"Type of ||| pred_disp: {type(pred_disp)}")
#     print(f"Shape of ||| pred_disp: {getattr(pred_disp, 'shape', None)}")
#     if not opt.no_eval:
#         evaluate_predictions(opt, pred_disps,ground_truths, MAX_DEPTH,MIN_DEPTH)

#     print("-> Done!")
def evaluate_predictions(opt,pred_disp, gt_depth, min_depth=1e-3, max_depth=80):
    """
    Evaluates a single prediction against ground truth depth.

    Args:
        pred_disp (numpy.ndarray): Predicted disparity map.
        gt_depth (numpy.ndarray): Ground truth depth map.
        opt: Options object containing evaluation configurations.
        min_depth (float): Minimum depth value for evaluation.
        max_depth (float): Maximum depth value for evaluation.

    Returns:
        tuple: (errors, scaled_pred_depth, scaling_ratio)
    """
    gt_height, gt_width = gt_depth.shape[:2]

    print(f"Type of pred_disp: {type(pred_disp)}")
    print(f"Shape of pred_disp: {getattr(pred_disp, 'shape', None)}")


    # Resize predicted disparity to ground truth dimensions
    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
    pred_depth = 1 / pred_disp

    if opt.eval_split == "eigen":
        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
    else:
        mask = gt_depth > 0

    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    # Apply scaling factor
    pred_depth *= opt.pred_depth_scale_factor

    scaling_ratio = None
    if not opt.disable_median_scaling:
        scaling_ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= scaling_ratio

    pred_depth = np.clip(pred_depth, min_depth, max_depth)

    # Compute error metrics
    errors = compute_errors(gt_depth, pred_depth)

    return errors, pred_depth, scaling_ratio

if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    #opts.load_weights_folder="C:\Users\bsef0\Downloads\MonoDepth\monodepth2-master\monodepth2-master\pothole_logs\pothole\models\weights_19"
    #opts.data_path= "C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset"

    opts.data_path ="C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New\\" #C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset\\splits"
    opts.load_weights_folder = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\monodepth2-master\\monodepth2-master\\pothole_logs\\pothole\\models\\weights_19"
    
    
    evaluate(opts)


