from pathlib import Path
import numpy as np
import os
import shutil

#set the source and destination folder paths
source_folder = 'data/segmented_data/segmentation_masks_tif'
destination_folder = 'data/segmented_data/segmentation_masks_tif_blurred'

#ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

#iterate through files in the source folder
for filename in os.listdir(source_folder):
    #check if the file ends with '_blurred.tif'
    if filename.endswith('_blurred_masks.tif'):
        #construct full file paths
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        #move the file
        shutil.move(source_file, destination_file)
        print(f'Moved: {filename}')


# #specify the folder path
# source = 'data/segmented_data/segmentations'
# folder_path = Path(source)

# #iterate through all files in the folder
# for item in folder_path.iterdir():
#    if item.is_file():  #check if it's a file (not a directory)
#        print(f"File: {item.name}")

#    #allow loading object arrays
#    file_path = f'{source}/{item.name}'
#    data = np.load(file_path, allow_pickle=True)  

#    #unpack the object
#    data = data.item()

#    #access the image data from the dictionary
#    image_data = data['masks']
#    image_data = image_data.astype(np.int32)

#    np.save(f'data/segmented_data/segmentation_masks/{item.name[0:-7]}mask.npy', image_data)

# '''i used this file whenever i had to transfer a large amount of files or iterate over directories'''
