import os
import glob
import shutil

# Source and destination folders
source_folder = "path/to/source/folder"
destination_folder = "path/to/destination/folder"

# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Loop through all files matching the pattern and move them
for file_path in glob.glob(os.path.join(source_folder, "*_seg.npy")):
    shutil.move(file_path, destination_folder)

print("Files moved successfully!")

# # Folder containing the files
# folder_path = "path/to/your/folder"

# # Loop through all files that match the pattern
# for file_path in glob.glob(os.path.join(folder_path, "*_editted_seg.npy")):
#     # New filename (replace '_editted_seg.npy' with '_seg.npy')
#     new_file_path = file_path.replace("_editted_seg.npy", "_seg.npy")
    
#     # Rename the file
#     os.rename(file_path, new_file_path)

# print("Renaming complete!")