import numpy as np
#import matplotlib.pyplot as plt


#allow loading object arrays
file_path = 'data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0000_seg.npy'
data = np.load(file_path, allow_pickle=True)

#unpack the object
data = data.item()

#access the image data from the dictionary (replace 'image_key' with the correct key)
image_data = data['outlines']

#fill in the cells using the outlines
#serialize the new numpy array into an .npy file with the corresponding file name




#greyscale image
#plt.imshow(image_data, cmap='gray')
#plt.axis('off')
#plt.show()