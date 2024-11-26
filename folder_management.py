from pathlib import Path
import numpy as np
import os
import shutil

# Set the source and destination folder paths
source_folder = 'data/segmented_data/segmentation_masks_tif'
destination_folder = 'data/segmented_data/segmentation_masks_tif_blurred'

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Iterate through files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file ends with '_blurred.tif'
    if filename.endswith('_blurred_masks.tif'):
        # Construct full file paths
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # Move the file
        shutil.move(source_file, destination_file)
        print(f'Moved: {filename}')



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
