from cellpose import models
import tifffile as tiff
import cv2
import numpy as np

# read in target image
image = tiff.imread('../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0001.tif')

# instantiate the cellpose model
model = models.Cellpose(model_type='cyto', gpu=True)

# unpack cellpose output
masks, flows, styles, diams = model.eval([np.array(image)], diameter=60, channels=[0, 0])

# combine dx and dy for each pixel into a list of [dx, dy]
gradient_vectors = np.stack((flows[0][1][0], flows[0][1][1]), axis=-1)

# flatten the 3D array (height, width, 2) to a 2D array where each element is [dx, dy]
gradient_vectors = gradient_vectors.reshape(1024,1024,2)

visualisation = np.array(image)

#skip every 15 pixels for visualisation 
step = 20

#loop through the image and draw arrows
for y in range(0, visualisation.shape[0], step):
    for x in range(0, visualisation.shape[1], step):
        #get the flow vectors at each (x, y)
        dx, dy = gradient_vectors[y][x][1], gradient_vectors[y][x][0]

        #start and end points for the arrow (moving from (x, y) to (x + dx, y + dy))
        start_point = (x, y)
        end_point = (int(x + 3*dx), int(y + 3*dy))

        #draw the arrow on the image (use arrowedLine for visualisation)
        cv2.arrowedLine(visualisation, start_point, end_point, (255, 0, 0), 1, tipLength=0.5)

#display the result
cv2.imshow("Cellpose Output with Arrows", visualisation)
cv2.waitKey(0)
cv2.destroyAllWindows()