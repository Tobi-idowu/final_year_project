import os

folder_path = "data/segmented_data/images"

# Get all file names recursively
file_names = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_names.append(os.path.join(root, file).split("/")[-1])

# Path to the folder containing the corresponding files
folder_path = "data/segmented_data/segmentations"

# Check if corresponding _seg.npy files exist
for file in file_names:
    base_name = os.path.splitext(file)[0]  # Remove .tif extension
    corresponding_file = f"{base_name}_seg.npy"
    
    if corresponding_file in os.listdir(folder_path):  
        print(f"✅ Found: {corresponding_file}")
    else:
        print(f"❌ Missing: {corresponding_file}")