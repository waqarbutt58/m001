# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from customTrainer import CustomTrainer
from options import MonodepthOptions
from datasets import CustomDataset
options = MonodepthOptions()
opts = options.parse()
opts.use_stereo = False
opts.frame_ids = [0]  # Monocular mode
opts.num_epochs=50
opts.learning_rate=1e-3

# if "pose_encoder" in opts.models_to_load:
#     opts.models_to_load.remove("pose_encoder")
#     print("Removed 'pose_encoder' from models_to_load.")
opts.models_to_load = [m for m in opts.models_to_load if m not in ["pose", "pose_encoder"]]


if __name__ == "__main__":
    trainer = CustomTrainer(opts)
    trainer.train()
