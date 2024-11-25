from cellpose import io, models, train

# Set your image and mask directories
image_dir = 'path/to/directory/'
mask_dir = 'path/to/directory/'

# Load the training data (images and masks)
# Ensure that the images and masks have the same naming convention
images, masks, _, _ = io.load_train_test_data(image_dir, mask_dir)

# Specify model parameters
model = models.Cellpose(model_type='cyto', gpu=True)  # or 'nuclei' for nuclear segmentation

# Train the model
train.train_model(images, masks, model=model, diameter=60, batch_size=8, epochs=100, augment=True)

#i then need to serialize the model