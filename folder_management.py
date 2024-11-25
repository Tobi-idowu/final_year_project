from pathlib import Path
import numpy as np
import os

# Path to the directory containing the files
mask_dir = "/home/oi5/final_year_project/data/segmented_data/segmentation_masks"

# Iterate through all files in the directory
for filename in os.listdir(mask_dir):
    # Check if the filename ends with "mask.tif"
    if filename.endswith("mask.npy") and not filename.endswith("masks.npy"):
        # Create the new filename
        new_filename = filename.replace("mask.npy", "masks.npy")
        
        # Get full paths
        old_path = os.path.join(mask_dir, filename)
        new_path = os.path.join(mask_dir, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")



# # Specify the folder path
# source = 'data/segmented_data/segmentations'
# folder_path = Path(source)

# # Iterate through all files in the folder
# for item in folder_path.iterdir():
#     if item.is_file():  # Check if it's a file (not a directory)
#         print(f"File: {item.name}")

#     # Allow loading object arrays
#     file_path = f'{source}/{item.name}'
#     data = np.load(file_path, allow_pickle=True)  

#     # Unpack the object
#     data = data.item()

#     # Access the image data from the dictionary
#     image_data = data['masks']
#     image_data = image_data.astype(np.int32)

#     np.save(f'data/segmented_data/segmentation_masks/{item.name[0:-7]}mask.npy', image_data)

# '''i used this file whenever i had to transfer a large amount of files or iterate over directories'''
