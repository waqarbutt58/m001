import os

def create_file_list(data_dir, output_file, subset):
    """
    Creates a file list for the given subset of the dataset.

    Args:
        data_dir (str): Path to the dataset root folder.
        output_file (str): Output text file to store file paths.
        subset (str): Subfolder for the dataset (e.g., "train" or "val").
    """
    subset_path = os.path.join(data_dir, subset)
    with open(output_file, 'w') as f:
        for root, _, files in os.walk(subset_path):
            for file in sorted(files):
                if file.endswith(('.jpg', '.png')):
                    relative_path = os.path.relpath(os.path.join(root, file), data_dir)
                    f.write(relative_path + '\n')
    print(f"{subset} file list saved to {output_file}")

# Define paths
data_dir = "data"  # Path to your dataset folder
train_output = "train_files.txt"
val_output = "val_files.txt"
test_output = "test_files.txt"

# Generate file lists
#create_file_list(data_dir, train_output, "trainA")
#create_file_list(data_dir, val_output, "valA")
create_file_list(data_dir, test_output, "testA")