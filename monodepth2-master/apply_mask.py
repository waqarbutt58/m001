import os
import cv2
import numpy as np


folder_path = "C:\\Users\\bsef0\\Downloads\\MonoDepth\\monodepth2-master\\Potholes - Copy\\"
  

#resize_images(folder_mask)

file_list = os.listdir(folder_path)
file_list.sort()  # Sort the files alphabetically

# Loop through the files and rename them sequentially

for index, filename in enumerate(file_list):

       
    name = f"{index}.png"
    print(name)





    # Check if the file is an image (you can add more image extensions if needed)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
        path  = os.path.join(folder_path, filename)
        im = cv2.imread(f"{folder_path}{filename}", cv2.IMREAD_UNCHANGED)

        #im = cv2.resize(im, (1224,370), interpolation=cv2.INTER_LINEAR)

        maskname = filename.replace("_depth.png","")
        maskname = maskname.replace(".","-")
        maskname = "./mask/" +  maskname + ".png"
        if os.path.exists(maskname) == False:
            # print(maskname)
            continue
        mask = cv2.imread(f"{maskname}",cv2.IMREAD_UNCHANGED)
        print(mask.shape)

        mask[mask > 0] = 1
        # mask = cv2.resize(mask, (im.shape[1], im.shape[0]))

        im[mask == 0] = 0
        #masked_image = cv2.bitwise_and(im, im, mask=mask)

        cv2.imwrite(f"./ntrainB/{filename}", im)

        print(f"Applied mask for {filename}")
        break

            
        
		