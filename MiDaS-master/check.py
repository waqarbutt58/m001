import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "MiDaS_small"
midas.to("cuda" if torch.cuda.is_available() else "cpu").eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Load and preprocess image
img = cv2.imread("C:\\Users\\bsef0\\Downloads\\Potholes Sample\\Plain\\frame_24973.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = transform(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# Predict depth
with torch.no_grad():
    # Check shape before passing to model
    # Remove extra dimension if needed
    if input_tensor.dim() == 5:
        input_tensor = input_tensor.squeeze(1)
    prediction = midas(input_tensor)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()
#depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
#depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)

# Normalize the depth map to 0-255
depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

# Convert to 8-bit unsigned integer (grayscale image)
depth_gray = depth_normalized.astype(np.uint8)

# Save as grayscale image
cv2.imwrite("plain_output_depth_gray.jpg", depth_gray)

# Optional: Display the image in a window (if not running headless)
cv2.imshow("Depth (Grayscale)", depth_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Save outputs
#cv2.imwrite("output_depth_colored.jpg", cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
