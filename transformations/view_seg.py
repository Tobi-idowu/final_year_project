import numpy as np
import matplotlib.pyplot as plt
#from scipy.ndimage import binary_fill_holes


# Allow loading object arrays
file_path = 'cellpose_all/pupa_1_stage_1_cropped_0000_seg.npy'
data = np.load(file_path, allow_pickle=True)  

# Unpack the object
data = data.item()

print(f"Keys in the dictionary: {data.keys()}")

# Access the image data from the dictionary and convert its datatype from float to integer
image_data = data['masks']
image_data = image_data.astype(np.int32)

# greyscale image
plt.imshow(image_data, cmap='gray')
plt.axis('off')
plt.show()



'''fill in the outline

# Create a filled mask
filled_mask = np.zeros_like(image_data)

# Assume unique values are integers within a known range
for value in range(1, np.max(image_data) + 1):
    # Create a binary mask for the current cell
    binary_mask = image_data == value

    # Fill the interior of the cell
    filled_cell = binary_fill_holes(binary_mask)

    # Add the filled cell back to the mask with its unique value
    filled_mask[filled_cell] = value

# greyscale image
plt.imshow(filled_mask, cmap='gray')
plt.axis('off')
plt.show()'''