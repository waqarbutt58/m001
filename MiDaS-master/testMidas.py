import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from midas.dpt_depth import DPTDepthModel  # or the appropriate model file
from midas.model_loader import load_model

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DPTDepthModel()  # Initialize the model structure


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
model.load_state_dict(torch.load('pothole_midas_model.pth', map_location=device), strict=False)

model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize to 384x384
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

# Load and preprocess the input image
image_path = 'C:\\Users\\bsef0\\Downloads\\Potholes Sample\\frame_01151.jpg'  # Replace with your image path
image = Image.open(image_path).convert("RGB")
input_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Make the prediction
with torch.no_grad():
    depth_map = model(input_image)

# Post-process depth map (if needed)
depth_map = depth_map.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

# Visualize the depth map
plt.imshow(depth_map, cmap='inferno')
plt.colorbar()
plt.title("Predicted Depth Map")
plt.show()
