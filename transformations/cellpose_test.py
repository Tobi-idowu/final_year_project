from cellpose import models
import tifffile as tiff
import matplotlib.pyplot as plt

# import torch
# print(torch.cuda.is_available())  # Should return True
# print(torch.cuda.get_device_name(0))  # Should print your GPU name

image = tiff.imread(f'../cellpose_all/pupa_1_stage_1_cropped_0001.tif')

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

# Combine dx and dy for each pixel into a list of [dx, dy]
gradient_vectors = np.stack((flows[0][1][0], flows[0][1][1]), axis=-1)

# Flatten the 3D array (height, width, 2) to a 2D array where each element is [dx, dy]
gradient_vectors = gradient_vectors.reshape(1024,1024,2)

image = Image.open('../cellpose_all/pupa_1_stage_1_cropped_0001.tif')

gray_image = np.array(image.convert("L"))

visualization = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

# Skip every 15 pixels for visualization 
step = 15

# Loop through the image and draw arrows
for y in range(0, gray_image.shape[0], step):
    for x in range(0, gray_image.shape[1], step):
        # Get the flow vectors at each (x, y)
        dx, dy = gradient_vectors[y][x][1], gradient_vectors[y][x][0]  # flow[0, y, x] is dx, flow[1, y, x] is dy

        # Start and end points for the arrow (moving from (x, y) to (x + dx, y + dy))
        start_point = (x, y)
        end_point = (int(x + dx), int(y + dy))

        # Draw the arrow on the image (use arrowedLine for visualization)
        cv2.arrowedLine(visualization, start_point, end_point, (0, 255, 0), 1, tipLength=0.1)

# Display the result
cv2.imshow("Optical Flow with Arrows", visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()