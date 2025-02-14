from cellpose import io, models, train
import os
from tifffile import imread
import numpy as np
from skimage.color import rgb2gray

#set your image and mask directories
image_dir = '/home/oi5/final_year_project/data/segmented_data/images'
mask_dir = '/home/oi5/final_year_project/data/segmented_data/segmentation_masks_tif'

#target shape for resizing
target_shape = (1024, 1024)

#load and process images
images = []
for filename in sorted(os.listdir(image_dir)):
    if filename.endswith('.tif'):
        filepath = os.path.join(image_dir, filename)
        img = imread(filepath)
        
        #convert to greyscale if RGB
        if len(img.shape) == 3:  #rGB image
            images.append(img)
            #print(f"Mask {filename}: shape {img.shape}")
            #img = rgb2gray(img)
        
        #resize to target shape if needed
        if img.shape != target_shape:
            continue
        
        img = np.expand_dims(img, axis=-1)
        images.append(img)
        #print(f"Mask {filename}: shape {img.shape}")

#load and process images
masks = []
for filename in sorted(os.listdir(mask_dir)):
    if filename.endswith('.tif'):
        filepath = os.path.join(mask_dir, filename)
        mask = imread(filepath)
        
        #resize to target shape
        if mask.shape != target_shape:
            continue
        
        masks.append(mask)
        #print(f"Mask {filename}: shape {mask.shape}")

#convert to numpy arrays
images = np.array(images)
masks = np.array(masks)

print(f"Loaded {len(images)} images and {len(masks)} masks.")

#specify model parameters
model = models.Cellpose(model_type='cyto3', gpu=True)

#train the model
#train.train_seg(model.net, train_data=images, train_labels=masks, diameter=60, batch_size=8, n_epochs=100, augment=True, save_path="trained_cellpose")
train.train_seg(model, train_data=images, train_labels=masks, n_epochs=100, save_path="trained_cellpose")

#save the trained model
print("Trained model was saved.")

#image_path = 'data/segmented_data/test'
#mask_path = 'data/segmented_data/test2'

#print(f"Testing specific files:\nImage: {image_path}\nMask: {mask_path}")
#images, masks, _, _ = io.load_train_test_data(image_path, mask_path, mask_filter="_masks")
#print("Data loaded successfully!")