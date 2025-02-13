from cellpose import models
import tifffile as tiff
import matplotlib.pyplot as plt

# import torch
# print(torch.cuda.is_available())  # Should return True
# print(torch.cuda.get_device_name(0))  # Should print your GPU name

image = tiff.imread(f'cellpose_all/pupa_1_stage_1_cropped_0001.tif')

#this might have to be cyto3
model = models.Cellpose(model_type='cyto', gpu=True)

#unsure of the average diameter (average cell diameter in pixels)
# masks → The segmented cell masks (each cell has a unique integer label).
# flows → A list containing various flow-related outputs:
#   flows[0] → The gradient (vector) flow field, which is the gradient mask where every pixel points toward the cell center.
#   flows[1] → The cell boundary map.
#   flows[2] → The Cellpose "heat map" representation.
# styles → The style vectors used for training.
# diams → The estimated average cell diameter.
masks, flows, styles, diams = model.eval([image], diameter=60, channels=[0, 0])

plt.imshow(masks[0], cmap='jet')
plt.colorbar()
plt.show()