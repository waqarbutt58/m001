import os
import shutil
import random

# Configuration
dataset_path_jpg = "datasets/Potholes/trainA"
dataset_path_png = "datasets/Potholes/trainB"
output_path = "data"
val_count = 100
test_count = 100

# Ensure the dataset directories exist for JPG files
train_dir_jpg = os.path.join(output_path, "trainA")
val_dir_jpg = os.path.join(output_path, "valA")
test_dir_jpg = os.path.join(output_path, "testA")

print(train_dir_jpg)
print(val_dir_jpg)
print(test_dir_jpg)
os.makedirs(train_dir_jpg, exist_ok=True)
os.makedirs(val_dir_jpg, exist_ok=True)
os.makedirs(test_dir_jpg, exist_ok=True)

# Ensure the dataset directories exist for PNG files
train_dir_png = os.path.join(output_path, "trainB")
val_dir_png = os.path.join(output_path, "valB")
test_dir_png = os.path.join(output_path, "testB")

os.makedirs(train_dir_png, exist_ok=True)
os.makedirs(val_dir_png, exist_ok=True)
os.makedirs(test_dir_png, exist_ok=True)

# Get all JPG files in the dataset directory

files_jpg = [f for f in os.listdir(dataset_path_jpg) if os.path.isfile(os.path.join(dataset_path_jpg, f))]

# Separate even and odd numbered files for JPG
#even_files_jpg = sorted([f for f in files_jpg if int(os.path.splitext(f)[0]) % 2 == 0])
even_files_jpg = sorted([
    f for f in files_jpg
    if os.path.splitext(f)[0].split('.')[0].isdigit()  # Extract the part before the decimal and validate
    and int(os.path.splitext(f)[0].split('.')[0]) % 2 == 0  # Check if it's even
])
#odd_files_jpg = sorted([f for f in files_jpg if int(os.path.splitext(f)[0]) % 2 != 0])
odd_files_jpg = sorted([
    f for f in files_jpg
    if os.path.splitext(f)[0].split('.')[0].isdigit()  # Extract the part before the decimal and validate
    and int(os.path.splitext(f)[0].split('.')[0]) % 2 != 0  # Check if it's even
])
# # Shuffle the files for randomness
random.shuffle(even_files_jpg)
random.shuffle(odd_files_jpg)

# Split files into train, val, and test sets
def split_files(file_list, val_count, test_count):
    val_files = file_list[:val_count]
    test_files = file_list[val_count:val_count + test_count]
    train_files = file_list[val_count + test_count:]
    return train_files, val_files, test_files

even_train_jpg, even_val_jpg, even_test_jpg = split_files(even_files_jpg, val_count, test_count)
odd_train_jpg, odd_val_jpg, odd_test_jpg = split_files(odd_files_jpg, val_count, test_count)

# Combine even and odd files for each dataset
train_files_jpg = even_train_jpg + odd_train_jpg
val_files_jpg = even_val_jpg + odd_val_jpg
test_files_jpg = even_test_jpg + odd_test_jpg

# Shuffle the combined datasets
random.shuffle(train_files_jpg)
random.shuffle(val_files_jpg)
random.shuffle(test_files_jpg)

# Move the JPG files to their respective directories
def move_files(file_list, dest_dir, src_path):
    for file in file_list:
        shutil.move(os.path.join(src_path, file), os.path.join(dest_dir, file))

move_files(train_files_jpg, train_dir_jpg, dataset_path_jpg)
move_files(val_files_jpg, val_dir_jpg, dataset_path_jpg)
move_files(test_files_jpg, test_dir_jpg, dataset_path_jpg)

# Corresponding PNG files
train_files_png = [f.replace('.jpg', '.png') for f in train_files_jpg]
val_files_png = [f.replace('.jpg', '.png') for f in val_files_jpg]
test_files_png = [f.replace('.jpg', '.png') for f in test_files_jpg]

# Move the PNG files to their respective directories
move_files(train_files_png, train_dir_png, dataset_path_png)
move_files(val_files_png, val_dir_png, dataset_path_png)
move_files(test_files_png, test_dir_png, dataset_path_png)

print("JPG and corresponding PNG files have been split and moved to train, val, test and trainB, valB, testB sets.")
