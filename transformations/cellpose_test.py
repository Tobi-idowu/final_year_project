from cellpose import models
import tifffile as tiff
import cv2
import numpy as np

#import torch
#print(torch.cuda.is_available())  #should return True
#print(torch.cuda.get_device_name(0))  #should print your GPU name

image = tiff.imread(f'../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0001.tif')

#this might have to be cyto3
model = models.Cellpose(model_type='cyto', gpu=True)

#unsure of the average diameter (average cell diameter in pixels)
#masks → The segmented cell masks (each cell has a unique integer label).
#flows → A list containing various flow-related outputs:
#  flows[0] → The gradient (vector) flow field, which is the gradient mask where every pixel points toward the cell center.
#  flows[1] → The cell boundary map.
#  flows[2] → The Cellpose "heat map" representation.
#styles → The style vectors used for training.
#diams → The estimated average cell diameter.
masks, flows, styles, diams = model.eval([image], diameter=60, channels=[0, 0])

#combine dx and dy for each pixel into a list of [dx, dy]
gradient_vectors = np.stack((flows[0][1][0], flows[0][1][1]), axis=-1)

#flatten the 3D array (height, width, 2) to a 2D array where each element is [dx, dy]
gradient_vectors = gradient_vectors.reshape(1024,1024,2)

visualisation = np.array(image)

#skip every 15 pixels for visualisation 
step = 15

#loop through the image and draw arrows
for y in range(0, visualisation.shape[0], step):
    for x in range(0, visualisation.shape[1], step):
        #get the flow vectors at each (x, y)
        dx, dy = gradient_vectors[y][x][1], gradient_vectors[y][x][0]  #flow[0, y, x] is dx, flow[1, y, x] is dy

        #start and end points for the arrow (moving from (x, y) to (x + dx, y + dy))
        start_point = (x, y)
        end_point = (int(x + dx), int(y + dy))

        #draw the arrow on the image (use arrowedLine for visualisation)
        cv2.arrowedLine(visualisation, start_point, end_point, (0, 255, 0), 1, tipLength=0.1)

#display the result
cv2.imshow("Optical Flow with Arrows", visualisation)
cv2.waitKey(0)
cv2.destroyAllWindows()