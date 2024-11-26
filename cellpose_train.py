from cellpose import io, models, train

# # Set your image and mask directories
# image_dir = '/home/oi5/final_year_project/data/segmented_data/images'
# mask_dir = '/home/oi5/final_year_project/data/segmented_data/segmentation_masks_tif'

# # Load the training data (images and masks)
# # Ensure that the images and masks have the same naming convention
# images, masks, _, _ = io.load_train_test_data(image_dir, mask_dir, mask_filter="_masks")

# # Specify model parameters
# model = models.Cellpose(model_type='cyto', gpu=True)

# # Train the model
# train.train_model(images, masks, model=model, diameter=60, batch_size=8, epochs=100, augment=True)

# # Save the trained model
# model.net.save_weights("trained_cellpose_model.pth")
# print("Trained model was saved.")

image_path = 'data/segmented_data/test'
mask_path = 'data/segmented_data/test2'

print(f"Testing specific files:\nImage: {image_path}\nMask: {mask_path}")
images, masks, _, _ = io.load_train_test_data(image_path, mask_path, mask_filter="_masks")
print("Data loaded successfully!")