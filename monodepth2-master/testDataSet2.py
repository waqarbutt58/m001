import torch
from torch.utils.data import DataLoader
from datasets.PotholeDataset2 import PotholeDataset2  # Make sure to use your actual path
from torchvision import transforms
import matplotlib.pyplot as plt
import time

def test_dataset():
    # Define some parameters for the test
    data_path = 'C:\\Users\\bsef0\\Downloads\\MonoDepth2_Dataset'  # Update path to your dataset
    filenames = ['1722417710.614009', '1722417390.990967']  # Add more filenames from your dataset
    height = 240  # Desired image height
    width = 320   # Desired image width
    batch_size = 4  # Number of samples per batch
    num_workers = 0  # For debugging, set to 0 to avoid multiprocessing issues

    # Create the dataset and dataloader
    dataset = PotholeDataset2(data_path, filenames, height, width, is_train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Check the length of the dataset
    print(f"Dataset size: {len(dataset)}")

    # Test a single batch
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        
        # Print out the shapes of each item in the batch
        for key, value in batch_data.items():
            print(f"{key}: {value.shape}")
        
        # Visualize the first sample of the batch
        if batch_idx == 0:
            img = batch_data[("color", 0, 0)].permute(1, 2, 0).numpy()  # Original image tensor
            img_aug = batch_data[("color_aug", 0, 0)].permute(1, 2, 0).numpy()  # Augmented image tensor
            
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            
            plt.subplot(1, 2, 2)
            plt.imshow(img_aug)
            plt.title("Augmented Image")
            
            plt.show()

        # Test how long it takes to load a few batches
        if batch_idx == 5:
            break

def test_data_loader_performance():
    # Same as above, but to measure performance
    data_path = 'C:/Users/bsef0/Downloads/MonoDepth2_Dataset'  # Update path to your dataset
    filenames = ['1722417710.614009', '1722417390.990967']  # Add more filenames from your dataset
    height = 240  # Desired image height
    width = 320   # Desired image width
    batch_size = 4  # Number of samples per batch
    num_workers = 4  # Set to a higher value for better performance

    dataset = PotholeDataset2(data_path, filenames, height, width, is_train=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    start_time = time.time()

    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx == 10:
            break  # Load 10 batches and measure time
        print(f"Batch {batch_idx} loaded in {time.time() - start_time:.2f} seconds")
        start_time = time.time()

if __name__ == '__main__':
    print("Testing Dataset...")
    test_dataset()  # Test data loading and visualization

    print("\nTesting DataLoader Performance...")
    test_data_loader_performance()  # Test dataloader performance (speed)
