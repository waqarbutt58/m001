import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.mono_dataset import PotholeDataset3   # Replace with your dataset class
from networks import ResnetEncoder, DepthDecoder  # Ensure correct imports
from utils import save_depth_image  # A utility function to save depth maps as images
from utils import *
from layers import *

def load_model(model_path):
    # Load your trained model
    encoder = ResnetEncoder(18, pretrained=False)  # Example, adapt to your model config
    decoder = DepthDecoder(encoder.num_ch_enc)
    model = torch.nn.ModuleDict({"encoder": encoder, "depth": decoder})

    # Load weights from the checkpoint
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model

import torch
from networks import ResnetEncoder, DepthDecoder

def load_model(model_dir, device="cuda"):
    """
    Load a pre-trained MonoDepth2 model from multiple saved components.
    
    :param model_dir: Path to the directory containing the saved .pth files.
    :param device: Device to load the model (default: "cuda" if available, else "cpu").
    :return: Dictionary containing the loaded model components.
    """

    # Determine device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Initialize model dictionary
    model = {}

    # Load encoder
    encoder_path = f"{model_dir}/encoder.pth"
    encoder_dict = torch.load(encoder_path, map_location=device)
    encoder = ResnetEncoder(18, pretrained=False)  # Adjust num_layers if different
    encoder.load_state_dict(encoder_dict, strict=False)
    encoder.to(device)
    encoder.eval()
    model["encoder"] = encoder

    # Load depth decoder
    depth_path = f"{model_dir}/depth.pth"
    depth_dict = torch.load(depth_path, map_location=device)
    depth_decoder = DepthDecoder(encoder.num_ch_enc)
    depth_decoder.load_state_dict(depth_dict, strict=False)
    depth_decoder.to(device)
    depth_decoder.eval()
    model["depth"] = depth_decoder

    # Load pose encoder (if needed)
    pose_encoder = None
    pose_encoder_path = f"{model_dir}/pose_encoder.pth"
    try:
        pose_encoder_dict = torch.load(pose_encoder_path, map_location=device)
        pose_encoder = ResnetEncoder(18, pretrained=False, num_input_images=2)  # Adjust if needed
        pose_encoder.load_state_dict(pose_encoder_dict, strict=False)
        pose_encoder.to(device)
        pose_encoder.eval()
        model["pose_encoder"] = pose_encoder
    except FileNotFoundError:
        print("Pose encoder not found, skipping...")

    # Load pose decoder (if needed)
    pose_decoder = None
    pose_path = f"{model_dir}/pose.pth"
    try:
        pose_decoder_dict = torch.load(pose_path, map_location=device)
        pose_decoder = DepthDecoder(pose_encoder.num_ch_enc, num_output_channels=6)  # 6-DoF pose
        pose_decoder.load_state_dict(pose_decoder_dict, strict=False)
        pose_decoder.to(device)
        pose_decoder.eval()
        model["pose_decoder"] = pose_decoder
    except FileNotFoundError:
        print("Pose decoder not found, skipping...")

    # Load optimizer state (useful for resuming training)
    optimizer_state = None
    adam_path = f"{model_dir}/adam.pth"
    try:
        optimizer_state = torch.load(adam_path, map_location=device)
        model["optimizer_state"] = optimizer_state
    except FileNotFoundError:
        print("Optimizer state not found, skipping...")

    print("Model successfully loaded!")
    return model


def evaluate(model, dataloader, device):
    model["encoder"].to(device)
    model["depth"].to(device)

    model["encoder"].eval()
    model["depth"].eval()
    
    total_loss = 0.0
    num_samples = 0
    results = []
    """Evaluate the depth model and print final averaged error metrics."""
    total_errors = np.zeros(10)  # To accumulate error metrics

    with torch.no_grad():
        for batch_idx, inputs in enumerate(dataloader):
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Get input image & ground truth depth
            image = inputs[("color", 0, 0)]
            depth_gt = inputs[('depth_gt', 0, 0)]

            # Forward pass through encoder and depth decoder
            features = model["encoder"](image)
            pred_depth = model["depth"](features)  # Returns a depth map

            # Get loss using total_depth_loss function
            #loss = model["encoder"].total_depth_loss(pred_depth[0], depth_gt, image)

            generate_images_pred_pothole(inputs,pred_depth)
            pred_depth_scale_0 = pred_depth[("depth", 0, 0)]
            loss = total_depth_loss(pred_depth_scale_0, depth_gt,image)

            print(pred_depth_scale_0.shape)

            # Convert tensors to numpy arrays
            pred_depth_np = pred_depth_scale_0[0].squeeze().cpu().numpy()
            depth_gt_np = depth_gt.squeeze().cpu().numpy()

            # Compute errors for this batch
            errors = compute_errors(depth_gt_np, pred_depth_np)
            # Accumulate errors and loss
            total_errors += np.array(errors)
            total_loss += loss.item()
            num_samples += 1
      # Compute the average errors
        # Average values
    avg_errors = total_errors / num_samples
    avg_loss = total_loss / num_samples

    print("\n==== Final Evaluation Metrics ====")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Absolute Relative Error: {avg_errors[0]:.4f}")
    print(f"Squared Relative Error: {avg_errors[1]:.4f}")
    print(f"RMSE: {avg_errors[2]:.4f}")
    print(f"Log RMSE: {avg_errors[3]:.4f}")
    print(f"Threshold a1 (1.25): {avg_errors[4]:.4f}")
    print(f"Threshold a2 (1.25^2): {avg_errors[5]:.4f}")
    print(f"Threshold a3 (1.25^3): {avg_errors[6]:.4f}")
    print(f"MAE: {avg_errors[7]:.4f}")
    print(f"iMAE: {avg_errors[8]:.4f}")
    print(f"iRMSE: {avg_errors[9]:.4f}")
    return results, avg_loss

def total_depth_loss(pred_depth, gt_depth, image, mask=None):
        """Total loss for depth estimation"""
        l1_loss = l1_depth_loss(pred_depth, gt_depth, mask)
        scale_inv_loss = scale_invariant_loss(pred_depth, gt_depth, mask)
        #smooth_loss = self.edge_aware_smoothness_loss(pred_depth, image)

        # Weighted sum of losses
        total_loss = 0.5 * l1_loss + 0.5 * scale_inv_loss #+ 0.1 * smooth_loss
        return total_loss
def l1_depth_loss(pred_depth, gt_depth, mask=None):
            """Computes L1 loss between predicted and ground truth depth"""
            if mask is not None:
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]
            return torch.mean(torch.abs(pred_depth - gt_depth))
        
        
def scale_invariant_loss(pred_depth, gt_depth, mask=None):
            """Computes scale-invariant loss for depth prediction"""
            if mask is not None:
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

            d = torch.log(pred_depth + 1e-6) - torch.log(gt_depth + 1e-6)
            loss = torch.mean(d ** 2) - (torch.mean(d) ** 2)
            return loss

def save_results(results, output_dir):
    # Save the results to disk (e.g., depth maps, images)
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (input_image, pred_depth, gt_depth) in enumerate(results):
        # Save predicted depth map
        save_depth_image(pred_depth, os.path.join(output_dir, f"pred_depth_{i}.png"))
        # Save ground truth depth map
        save_depth_image(gt_depth, os.path.join(output_dir, f"gt_depth_{i}.png"))

def generate_images_pred_pothole(inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        scales=[0,1,2,3]
        frame_ids=[0]
        v1_multiscale=True
        for scale in scales:
            disp = outputs[("disp", scale)]  # Retrieve the disparity at the current scale
            
            if v1_multiscale:
                source_scale = scale
            else:
                # Resize the disparity map to the target image size
                disp = F.interpolate(
                    disp, [224, 320], mode="bilinear", align_corners=False)
                source_scale = 0
            
            # Convert disparity to depth
            _, depth = disp_to_depth(disp, 0.1, 100)

            # Save the depth map at the current scale
            outputs[("depth", 0, scale)] = depth

            # Handle frame re-projection (optional, depending on use case)
            for i, frame_id in enumerate(frame_ids[1:]):
                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                
                

def main():
    # Configuration
    split ="C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New\\splits\\pothole\\"
    model_path = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\monodepth2-master\\monodepth2-master\\pothole_logs\\pothole\\models\\weights_18"
    test_data_path = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New\\"
    output_dir = "output/evaluation_results"
    fpath = os.path.join(os.path.dirname(__file__), "splits", split, "{}_files.txt")
    print(fpath)
    test_filenames = readlines(fpath.format("test"))

    # Prepare Dataset and DataLoader
        
    test_dataset = PotholeDataset3(test_data_path,test_filenames,224,320,is_train=False)  # Replace with your dataset class
    test_loader = DataLoader(test_dataset, 8, True,
            num_workers=12, pin_memory=True, drop_last=True)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path)
    
    # Evaluate the model
    results, avg_loss = evaluate(model, test_loader, device)

    # Save results (depth maps)
    save_results(results, output_dir)
import numpy as np

# def compute_errors(gt, pred):
#     """Compute error metrics between ground truth and predicted depths."""
#     gt = np.clip(gt, 1e-3, None)  # Avoid division by zero (assuming depth can't be zero)
#     pred = np.clip(pred, 1e-3, None)

#     thresh = np.maximum(gt / pred, pred / gt)
#     a1 = (thresh < 1.25).mean()
#     a2 = (thresh < 1.25 ** 2).mean()
#     a3 = (thresh < 1.25 ** 3).mean()

#     rmse = np.sqrt(np.mean((gt - pred) ** 2))

#     rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))

#     abs_rel = np.mean(np.abs(gt - pred) / gt)

#     sq_rel = np.mean(((gt - pred) ** 2) / gt)

#     return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
import numpy as np

def compute_errors(gt, pred):
    """Compute error metrics between ground truth and predicted depths."""
    gt = np.clip(gt, 1e-3, None)
    pred = np.clip(pred, 1e-3, None)

    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    mae = np.mean(np.abs(gt - pred))

    inv_gt = 1.0 / gt
    inv_pred = 1.0 / pred
    imae = np.mean(np.abs(inv_gt - inv_pred))
    irmse = np.sqrt(np.mean((inv_gt - inv_pred) ** 2))

    return [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, mae, imae, irmse]



if __name__ == "__main__":
    main()
