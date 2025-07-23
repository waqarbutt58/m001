from midas import transforms
from midas.model_loader import load_model
from midas.transforms import  Resize, NormalizeImage, PrepareForNet
from torch.utils.data import DataLoader
from PotholeDataset import PotholeDataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

# Define transformations
transform = transforms.Compose([
    Resize(384, 384),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
# Load datasets
tarin_images_path="C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\midas_dataset\\train\\images"
tarin_images_depth="C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\midas_dataset\\train\\depth_maps"
val_images_path="C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\midas_dataset\\val\\images"
val_images_depth="C:\\Users\\bsef0\\Downloads\\MonoDepth\\MiDaS-master\\MiDaS-master\\midas_dataset\\val\\depth_maps"

train_dataset = PotholeDataset(tarin_images_path, tarin_images_depth, transform=transform)
val_dataset = PotholeDataset(val_images_path, val_images_depth, transform=transform)


# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)



# Set paths and model configuration
device = "cuda"  # or "cpu"
model_path = "weights/dpt_beit_large_512.pt"  # Adjust the path to your weights file
model_type = "dpt_beit_large_512"  # Ensure this matches the weights
# Load the model
model, transform, net_w, net_h = load_model(
    device=device,
    model_path=model_path,
    model_type=model_type
)

print("Model loaded successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.MSELoss()  # Use Mean Squared Error for depth maps
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 1
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    total_batches = len(train_loader)

    for batch_idx, (images, depths) in enumerate(train_loader, 1):  # Start index at 1
        images = images.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        #print("Outputs shape:", outputs.shape)
        #print("Depths shape:", depths.shape)
        outputs = outputs.unsqueeze(1)  # Add a channel dimension
        #print("New Outputs shape:", outputs.shape)  # Should be (8, 1, 384, 384)
        depths = depths[:, 0:1, :, :]  # Keep only the first channel

        loss = criterion(outputs, depths)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Processed {batch_idx}/{total_batches} batches", end="\r")
        if batch_idx == 5:  # Stop after the 10th batch  
            break  

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (images, depths) in enumerate(val_loader, 1):  # Start index at 1 images, depths in val_loader:
            images = images.to(device)
            depths = depths.to(device)

            outputs = model(images)
            outputs = outputs.unsqueeze(1)  # Add a channel dimension
            depths = depths[:, 0:1, :, :]  # Keep only the first channel
            loss = criterion(outputs, depths)
            val_loss += loss.item()
            print(f"Processed {batch_idx}/{total_batches} batches", end="\r")
            if batch_idx == 2:  # Stop after the 10th batch  
                break  

        print(f"Validation Loss: {val_loss/len(val_loader)}")

torch.save(model.state_dict(), "pothole_midas_model.pth")
