import numpy as np
import matplotlib.pyplot as plt


#for i in range(10):

# Allow loading object arrays
file_path = 'cellpose_all/pupa_1_stage_1_cropped_0000_seg.npy'
data = np.load(file_path, allow_pickle=True)  

# Now, let's inspect what the object contains
#print(f"Data content after unpacking: {data}")

# Unpack the object
data = data.item()

print(f"Keys in the dictionary: {data.keys()}")

# Access the image data from the dictionary (replace 'image_key' with the correct key)
image_data = data['outlines']

# Grayscale image
plt.imshow(image_data, cmap='gray')
plt.axis('off')
plt.show()