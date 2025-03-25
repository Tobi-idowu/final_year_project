import os

# Define folder paths
tif_folder = "data/segmented_data/images/rebecca_images"  # Contains original .npy files
seg_folder = "data/segmented_data/segmentations"  # Should contain corresponding _seg.npy files

# Get all .tif filenames (without extensions)
tif_files = {os.path.splitext(f)[0] for f in os.listdir(tif_folder) if f.endswith(".tif")}

# Get all _seg.npy filenames (remove the _seg.npy part)
seg_files = {f[:-8] for f in os.listdir(seg_folder) if f.endswith("_seg.npy")}  # Removes "_seg.npy"

# Find missing files
missing_files = tif_files - seg_files

# Report results
if missing_files:
    print("The following .tif files do not have corresponding _seg.npy files:")
    for file in missing_files:
        print(f"{file}.tif")
else:
    print("All .tif files have corresponding _seg.npy files.")

# # Source and destination folders
# source_folder = "path/to/source/folder"
# destination_folder = "path/to/destination/folder"

# # Ensure the destination folder exists
# os.makedirs(destination_folder, exist_ok=True)

# # Loop through all files matching the pattern and move them
# for file_path in glob.glob(os.path.join(source_folder, "*_seg.npy")):
#     shutil.move(file_path, destination_folder)

# print("Files moved successfully!")

# # Folder containing the files
# folder_path = "path/to/your/folder"

# # Loop through all files that match the pattern
# for file_path in glob.glob(os.path.join(folder_path, "*_editted_seg.npy")):
#     # New filename (replace '_editted_seg.npy' with '_seg.npy')
#     new_file_path = file_path.replace("_editted_seg.npy", "_seg.npy")
    
#     # Rename the file
#     os.rename(file_path, new_file_path)

# print("Renaming complete!")