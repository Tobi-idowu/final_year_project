from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path = 'cellpose_all/pupa_1_stage_1_cropped_0000_seg.npy'
data = np.load(file_path, allow_pickle=True)

# Unpack the object
data = data.item()

# Access the image data from the dictionary (replace 'image_key' with the correct key)
image_data = data['outlines']

np.set_printoptions(threshold=np.inf)
#print(image_data)

#image_data[image_data > 0] = 1
binary_image = (image_data <= 0).astype(np.uint8)

# Compute the distance transform
dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)  # L2 is Euclidean distance

# Display the result using matplotlib
plt.imshow(dist_transform, cmap='jet', vmin=0, vmax=30)  # Using a color map to visualize distances
plt.colorbar()
plt.show()

reconstruct = (dist_transform < 1)

# Grayscale image
plt.imshow(reconstruct, cmap='gray')
plt.axis('off')
plt.show()