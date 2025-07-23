from PIL import Image
import torch
from torchvision import transforms
import os

class PotholeDataset2(torch.utils.data.Dataset):
    def __init__(self, data_path, filenames, height, width, is_train=True):
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train

        # Define standard image transformations
        self.to_tensor = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
        ])
        
        # Define augmentation transformations
        self.augment = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
        ]) if is_train else self.to_tensor

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the image and depth map paths from the space-separated filenames in train_files.txt
        img_path, depth_map_path = self.filenames[idx].split()

        # Construct the full paths for image and depth map
        img_path = os.path.join(self.data_path, img_path)
        depth_map_path = os.path.join(self.data_path, depth_map_path)
        #img_path = f"{self.data_path}/images/{self.filenames[idx]}.jpg"
        #depth_path = f"{self.data_path}/depth_maps/{self.filenames[idx]}_depth.png"

        # Load images
        image = Image.open(img_path).convert("RGB")
        depth_map = Image.open(depth_map_path)

        # Apply transformations
        image_tensor = self.to_tensor(image)
        depth_tensor = self.to_tensor(depth_map)

        # Augment color images
        image_aug = self.augment(image)

        # Return in expected format
        return {
            ("color", 0, 0): image_tensor,
            ("color_aug", 0, 0): image_aug,
            ("depth", 0, 0): depth_tensor/1000.0,
        }

# Example usage
# train_path="C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset\\splits\\pothole\\train_files.txt"
# # val_path="C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset\\splits\\pothole\\val_files.txt"
# # test_path="C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset\\splits\\pothole\\test_files.txt"
# dataset = PotholeDataset2("C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset",train_path,320,240)
# print(f"Total samples: {len(dataset)}")
