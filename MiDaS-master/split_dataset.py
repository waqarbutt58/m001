

import os
import shutil
import random

# Paths
dataset_dir = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New"  # Path to your original dataset

image_dir = os.path.join(dataset_dir, "images")
depth_dir = os.path.join(dataset_dir, "depth_map")

output_dir = "midas_dataset"  # Output directory
train_images_dir = os.path.join(output_dir, "train", "images")
train_depth_dir = os.path.join(output_dir, "train", "depth_maps")
val_images_dir = os.path.join(output_dir, "val", "images")
val_depth_dir = os.path.join(output_dir, "val", "depth_maps")

# Create output directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_depth_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_depth_dir, exist_ok=True)
#print(depth_dir)

# Get a list of images and depth maps
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith("_depth.png")])

# Filter only matching pairs
image_base_names = {os.path.splitext(f)[0] for f in image_files}
depth_base_names = {os.path.splitext(f)[0].replace("_depth", "") for f in depth_files}

# Find common files
common_base_names = image_base_names.intersection(depth_base_names)

# Get the matching image and depth map filenames
image_files = [f"{name}.jpg" for name in common_base_names]
depth_files = [f"{name}_depth.png" for name in common_base_names]

# Shuffle and split dataset manually
dataset = list(zip(image_files, depth_files))
random.seed(42)  # For reproducibility
random.shuffle(dataset)

train_size = int(0.8 * len(dataset))  # 80% for training
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]

# Move files to corresponding directories
def move_files(dataset, src_image_dir, src_depth_dir, dest_image_dir, dest_depth_dir):
    for image_file, depth_file in dataset:
        shutil.copy(os.path.join(src_image_dir, image_file), os.path.join(dest_image_dir, image_file))
        shutil.copy(os.path.join(src_depth_dir, depth_file), os.path.join(dest_depth_dir, depth_file))

# Distribute images and depth maps
move_files(train_dataset, image_dir, depth_dir, train_images_dir, train_depth_dir)
move_files(val_dataset, image_dir, depth_dir, val_images_dir, val_depth_dir)

print(f"Dataset has been prepared and saved to: {output_dir}")
print(f"Train images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
