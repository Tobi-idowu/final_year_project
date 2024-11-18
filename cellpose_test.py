from cellpose import models
import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np

#for i in range(1,10):
image = tiff.imread(f'cellpose_all/pupa_1_stage_1_cropped_0001.tif')

print(image.shape)

#model = models.Cellpose(model_type='cyto', gpu=True)
model = models.Cellpose(model_type='cyto')

#images = []
#io.imsave('cellpose_all/pupa_1_stage_1_cropped_0000.tif', images)
#images = io.load_images_labels('cellpose_all/pupa_1_stage_1_cropped_0000.tif')

#unsure of the average diameter (average cell diameter in pixels)
#it needs to account for the fact that cellpose will rescale the images from 1024 by 1024 to 512 by 512
#just divide by 2
masks, flows, styles, diams = model.eval([image], diameter=60, channels=[0, 0])

plt.imshow(masks[0], cmap='jet')
plt.colorbar()
plt.show()

'''Y, X = np.mgrid[0:1024, 0:1024]  # Y, X are row, column indices

# Extract the flow components: horizontal (dx) and vertical (dy)
dx = flows[:, :, 0]  # Horizontal component
dy = flows[:, :, 1]  # Vertical component

# Display the flow field using quiver plot
plt.figure(figsize=(6, 6))

# The quiver function plots arrows for the flow at each (X, Y) position
plt.quiver(X, Y, dx, dy, scale=10, color='r')  # Adjust scale to control arrow size
plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
plt.title("Optical Flow Field Visualization")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()'''