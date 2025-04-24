import numpy as np
import matplotlib.pyplot as plt

# read in the numpy object
file_path = '../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0000_seg.npy'
data = np.load(file_path, allow_pickle=True)  

# unpack the object
data = data.item()

# access the image data from the dictionary and convert its datatype from float to integer
image_data = data['masks']
image_data = image_data.astype(np.int32)

# display the cell masks (each cell is given a different integer value)
plt.imshow(image_data, cmap='gray')
plt.axis('off')
plt.show()