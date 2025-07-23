import json
import numpy as np
import cv2
import os
import skimage
import shutil


def create_mask(image_info, annotations, output_folder):
    # Create an empty mask as a numpy array
    mask_np = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    # Counter for the object number
    object_number = 1

    filename = image_info['file_name']
    filename = filename.split("_rgb")[0];
    file_depth = "{filename}_depth.png";
    image_depth = cv2.imread(file_depth)
	
    for ann in annotations:
        if ann['image_id'] == image_info['id']:
            # Extract segmentation polygon
            for seg in ann['segmentation']:
                # Convert polygons to a binary mask and add it to the main mask
                rr, cc = skimage.draw.polygon(seg[1::2], seg[0::2], mask_np.shape)
                
                mask_np[rr, cc] = 255
                print(image_depth[rr,cc])
                object_number += 1 #We are assigning each object a unique integer value (labeled mask)

    # Save the numpy array as a TIFF using tifffile library

    filename = f"{filename}.png"
    mask_path = os.path.join(output_folder, filename)
    cv2.imwrite(mask_path, mask_np)

    print(f"Saved mask for {image_info['file_name']} to {mask_path} having objects: {object_number}")

# Load and parse the JSON file
with open('portholes_annotations.json', 'r') as file:
    data = json.load(file)
    images = data['images']
    annotations = data['annotations']
    print(len(images), len(annotations))

    for im in images:
        create_mask(im, annotations, "mask")
        

