import cv2
import numpy as np
import matplotlib.pyplot as plt

# read in target image
file_path = '../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0000_seg.npy'
data = np.load(file_path, allow_pickle=True)

#unpack the object
data = data.item()

#access the image data from the dictionary (replace 'image_key' with the correct key)
image_data = data['outlines']

#convert to segmentation map (1s and 0s)
binary_image = (image_data <= 0).astype(np.uint8)

#compute the distance transform
dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)  #l2 is Euclidean distance

#display the result using matplotlib
plt.imshow(dist_transform, cmap='jet', vmin=0, vmax=30)  #using a color map to visualise distances
plt.colorbar()
plt.show()

#reconstruct the outlines from the distance transform
reconstruct = (dist_transform < 1)

#greyscale image
plt.imshow(reconstruct, cmap='gray')
plt.axis('off')
plt.show()