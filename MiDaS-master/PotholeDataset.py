from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np

class PotholeDataset(Dataset):
    def __init__(self, image_dir, depth_dir, transform=None):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.depths = sorted(os.listdir(depth_dir))

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        
        image_path = os.path.join(self.image_dir, self.images[index])
        depth_path = os.path.join(self.depth_dir, self.depths[index])

        # Load images
        image = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert("L")

        # Convert PIL to NumPy array
        image = np.array(image)
        depth = np.array(depth)

        # Convert depth to 3-channel grayscale (repeat across 3 channels)
        depth = np.stack([depth] * 3, axis=-1)  # Shape: (H, W) → (H, W, 3

        # Apply transformations
        #image = self.transform(image)  # Ensure this includes `ToTensor()`
        #depth = self.transform(depth)  # Ensure depth is also converted

        # Create a dummy mask (black image of the same size)
        mask = np.zeros_like(image[:, :, 0])  # Create a grayscale mask with zeros

        # Apply transformations
        transformed = self.transform({"image": image, "mask": mask})  # Include mask
        image = transformed["image"]
        
        # Apply MiDaS transformation to the depth image
        transformed_depth = self.transform({"image": depth, "mask": mask})
        depth = transformed_depth["image"]
        # Convert PIL to NumPy or PyTorch Tensor if needed
        # if isinstance(image, torch.Tensor):
        #     pass  # Already a tensor
        # else:
        #     image = torch.tensor(np.array(image))

        return image, depth

#dataset = PotholeDataset(root_dir='C:\\Users\\bsef0\\Downloads\\MonoDepth\\monodepth2-master\\datasets\\RawDataSets', transform=transform)