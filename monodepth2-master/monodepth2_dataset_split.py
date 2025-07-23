import os
import random

# Define dataset paths
dataset_root = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\dataset\\Potholes_Raw_New"#C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset"
image_folder = os.path.join(dataset_root, "images")
depth_folder = os.path.join(dataset_root, "depth_maps")

# Create splits directory
split_dir = os.path.join(dataset_root, "splits", "pothole")
os.makedirs(split_dir, exist_ok=True)

# List all image files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg")])
random.shuffle(image_files)  # Shuffle dataset

# Split data (80% train, 10% val, 10% test)
train_size = int(0.8 * len(image_files))
val_size = int(0.1 * len(image_files))

train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

# Write file paths to text files (relative paths)
def write_file_list(filename, file_list):
    with open(filename, "w") as f:
        for img in file_list:
            base_name = os.path.splitext(img)[0]  # Remove .jpg extension
            f.write(f"images/{img} depth_maps/{base_name}_depth.png\n")

write_file_list(os.path.join(split_dir, "train_files.txt"), train_files)
write_file_list(os.path.join(split_dir, "val_files.txt"), val_files)
write_file_list(os.path.join(split_dir, "test_files.txt"), test_files)

print("✅ Dataset successfully split into train, val, and test sets!")
