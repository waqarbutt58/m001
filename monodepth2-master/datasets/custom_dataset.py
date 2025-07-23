# from PIL import Image
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class CustomDataset(Dataset):
#     def __init__(self, data_path, filenames, height, width, is_train=False):
#         super(CustomDataset, self).__init__()
#         self.data_path = data_path
#         self.filenames = filenames
#         self.height = height
#         self.width = width
#         self.is_train = is_train

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, index):
#         inputs = {}
#         filename = self.filenames[index]

#         # Load image
#         img_path = os.path.join(self.data_path, filename)
#         img = Image.open(img_path).convert('RGB')
#         print(f"Image path: {img_path}")
#         if not os.path.exists(img_path):
#             raise FileNotFoundError(f"Image file does not exist: {img_path}")


#         img = Image.open(img_path).convert('RGB')
#         #img = img.resize((self.width, self.height), Image.ANTIALIAS)
#         image = image.resize(self.full_res_shape, Image.Resampling.LANCZOS)
#         inputs["color"] = torch.from_numpy(np.array(img).astype(np.float32) / 255).permute(2, 0, 1)

#         # Example: Add a dummy disparity map (optional)
#         # inputs["disparity"] = None

#         return inputs

from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, filenames, height, width, is_train=False):
        super(CustomDataset, self).__init__()
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.full_res_shape = (width, height)  # Define full resolution shape

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        filename = self.filenames[index]

        # Load image
        img_path = os.path.join(self.data_path, filename)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file does not exist: {img_path}")

        img = Image.open(img_path).convert('RGB')
        print(f"Image loaded from path: {img_path}")

        # Resize the image
        img = img.resize(self.full_res_shape, Image.Resampling.LANCZOS)

        # Convert the image to a tensor and normalize
        inputs["color"] = torch.from_numpy(np.array(img).astype(np.float32) / 255).permute(2, 0, 1)

        # Example: Add a dummy disparity map (optional)
        # inputs["disparity"] = None

        return inputs
