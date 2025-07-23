# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
from datasets.mono_dataset import PotholeDataset3
from torch.utils.data import DataLoader
from utils import *
import os
options = MonodepthOptions()
opts = options.parse()
opts.use_stereo = False
opts.frame_ids = [0]  # Monocular mode
opts.learning_rate=1e-3



opts.data_path = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New"#"C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset"
opts.split ="C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New\\splits\\pothole"#"c:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\splits"
opts.dataset ="pothole"
opts.width= 320 
opts.height= 224 
opts.batch_size =8
opts.num_epochs =20 
opts.learning_rate =1e-4 
opts.log_dir ="pothole_logs" 
opts.load_weights_folder = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\monodepth2-master\\monodepth2-master\\networks\\mono_640x192"
opts.log_frequency=100
# if "pose_encoder" in opts.models_to_load:
#     opts.models_to_load.remove("pose_encoder")
#     print("Removed 'pose_encoder' from models_to_load.")
opts.models_to_load = [m for m in opts.models_to_load if m not in ["pose", "pose_encoder"]]


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train_potholes()

# if __name__ == "__main__":
#     fpath = os.path.join(os.path.dirname(__file__), "splits", opts.split, "{}_files.txt")

#     train_filenames = readlines(fpath.format("train"))
#     dataset = PotholeDataset3(
#                 opts.data_path, train_filenames, opts.height, opts.width,
#                 is_train=True  # Keep only parameters your dataset supports
#             )

    # train_loader = DataLoader(
    #             dataset, opts.batch_size, True,
    #             num_workers=opts.num_workers, pin_memory=True, drop_last=True)
    # for i, data in enumerate(train_loader):
    #     depth = data[("depth", 0, 0)]
    #     print(f"Depth map {i} shape:", depth.shape)  # Should be (B, 1, H, W)
    #     print("Min depth:", depth.min(), "Max depth:", depth.max())
    # for batch in train_loader:
    #     color = batch["color"]  # RGB Image
    #     depth_gt = batch["depth_gt"]  # Ground Truth Depth Map
    #     K = batch["K"]  # Camera Intrinsics

    #     print("Color Image Shape:", color.shape)   # Expected: (B, 3, H, W)
    #     print("Depth Shape:", depth_gt.shape)      # Expected: (B, 1, H, W)
    #     print("Camera Intrinsics:", K)             # Expected: (B, 3, 3)
    #     break

