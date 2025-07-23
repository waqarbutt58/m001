import os
import random
import numpy as np
from PIL import Image  # Using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):
    """Load an image file as a PIL.Image in RGB format."""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')  # Ensure image is in RGB format


class PotholeDataset(data.Dataset):
    """Dataset class for loading RGB images."""

    def __init__(self, data_path, filenames, height, width, num_scales, is_train=False, img_ext='.jpg',frame_idxs=None, load_depth=False, brightness=(0.2, 0.5), contrast=(0.2, 0.5), saturation=(0.2, 0.5), hue=(-0.1, 0.1)):
        """
        Args:
            data_path (str): Root directory of the dataset.
            filenames (list): List of image file names (relative paths).
            height (int): Target height of the images.
            width (int): Target width of the images.
            num_scales (int): Number of scales for image resizing.
            is_train (bool): Whether this is a training dataset.
            img_ext (str): Image file extension (e.g., '.jpg').
        """
        super(PotholeDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext
        self.frame_idxs = frame_idxs if frame_idxs is not None else [0]
        self.load_depth = load_depth
        # Color augmentation parameters
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        # Default camera intrinsic matrix K (you can replace these values with your actual intrinsics)
        self.K = np.array([
            [self.width, 0, self.width / 2],
            [0, self.height, self.height / 2],
            [0, 0, 1]
        ])

        # Define augmentations for training and validation
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.resize = {i: transforms.Resize((self.height // (2 ** i), self.width // (2 ** i))) for i in range(num_scales)}
        
    def preprocess(self, inputs, color_aug):
        """Preprocess and augment input images."""
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                if i == -1:  # Base scale, use original image
                    frame = inputs[(n, im, -1)]  # Get the base image
                    inputs[(n, im, 0)] = self.resize[0](frame)  # Resize to target dimensions
                else:  # Resizing for other scales
                    frame = inputs[(n, im, i - 1)]  # Get the image for the previous scale
                    inputs[(n, im, i)] = self.resize[i](frame)  # Resize to the new scale

                # Convert image to tensor
                inputs[(n, im, i)] = self.to_tensor(inputs[(n, im, i)])

                # Apply color augmentation (if any)
                aug_img = color_aug(inputs[(n, im, i)])  # Apply augmentation on the image


                    # Ensure the augmented image is a tensor (not a PIL Image)
                if isinstance(aug_img, torch.Tensor):
                    inputs[(n + "_aug", im, i)] = aug_img  # Already a tensor, so no need for further conversion
                else:
                    # Convert to PIL Image and then to Tensor
                    inputs[(n + "_aug", im, i)] = self.to_tensor(aug_img)

    

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary."""
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjust intrinsics for each scale
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            # color_aug = transforms.ColorJitter.get_params(
            #     self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue
                )
        else:
            color_aug = (lambda x: x)
        
        # Apply the color augmentation to the loaded color image
        for i in self.frame_idxs:
            if ("color", i, -1) in inputs:
                inputs[("color_aug", i, 0)] = color_aug(inputs[("color", i, -1)])

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]


        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        # Ensure all returned images are tensors
        for k in list(inputs):
            if isinstance(inputs[k], Image.Image):  # If any image is still in PIL format
                inputs[k] = self.to_tensor(inputs[k])  # Convert to tensor

        return inputs
    
    def get_color(self, folder, frame_index, side=None, do_flip=False ,scale=0):
        """Load a color image, resize, and apply any transformations."""
        # Build the full image path
        #img_path = os.path.join(self.data_path, folder, f"{frame_index}.jpg")
        frame_index_str = str(frame_index).split('.')[0]
        # Build the full image path
        img_path = os.path.join(self.data_path, folder)#, f"{frame_index_str}.jpg")

        # Open the image
        img = Image.open(img_path).convert('RGB')

        # Apply flip augmentation if needed
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Resize the image
        img = self.resize[scale](img)

        return img

