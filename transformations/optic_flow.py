from PIL import Image
import cv2
import numpy as np

frame1 = Image.open('../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0006.tif')
frame2 = Image.open('../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0007.tif')

grey1 = np.array(frame1.convert("L"))
grey2 = np.array(frame2.convert("L"))

if grey1.shape != grey2.shape:
    raise ValueError("Images must be the same size for optical flow computation")

flow = cv2.calcOpticalFlowFarneback(
    grey1, grey2, None, pyr_scale=0.5,
    levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0
)

flow_visualisation = cv2.cvtColor(grey1, cv2.COLOR_GRAY2BGR)

#skip every 20 pixels for visualisation (you can adjust this number)
step = 20

#loop through the image and draw arrows
for y in range(0, grey1.shape[0], step):
    for x in range(0, grey1.shape[1], step):
        #get the flow vectors at each (x, y)
        flow_at_pixel = flow[y][x]
        
        #the flow is given as (dx, dy), where:
        dx, dy = flow_at_pixel[0], flow_at_pixel[1]

        #draw an arrow from (x, y) to (x + dx, y + dy)
        start_point = (x, y)
        end_point = (int(x + dx), int(y + dy))

        #draw the arrow on the image (use arrowedLine for visualisation)
        cv2.arrowedLine(flow_visualisation, start_point, end_point, (0, 255, 0), 1, tipLength=0.1)

#display the result
cv2.imshow("Optical Flow with Arrows", flow_visualisation)
cv2.waitKey(0)
cv2.destroyAllWindows()