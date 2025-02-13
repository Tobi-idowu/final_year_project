from PIL import Image
import cv2
import numpy as np

frame1 = Image.open('../cellpose_all/pupa_1_stage_1_cropped_0006.tif')
frame2 = Image.open('../cellpose_all/pupa_1_stage_1_cropped_0007.tif')

gray1 = np.array(frame1.convert("L"))
gray2 = np.array(frame2.convert("L"))

if gray1.shape != gray2.shape:
    raise ValueError("Images must be the same size for optical flow computation")

flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None, pyr_scale=0.5,
    levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0
)

flow_visualization = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)

# Skip every 20 pixels for visualization (you can adjust this number)
step = 20

# Loop through the image and draw arrows
for y in range(0, gray1.shape[0], step):
    for x in range(0, gray1.shape[1], step):
        # Get the flow vectors at each (x, y)
        flow_at_pixel = flow[y][x]
        
        # The flow is given as (dx, dy), where:
        dx, dy = flow_at_pixel[0], flow_at_pixel[1]

        # Draw an arrow from (x, y) to (x + dx, y + dy)
        start_point = (x, y)
        end_point = (int(x + dx), int(y + dy))

        # Draw the arrow on the image (use arrowedLine for visualization)
        cv2.arrowedLine(flow_visualization, start_point, end_point, (0, 255, 0), 1, tipLength=0.1)

# Display the result
cv2.imshow("Optical Flow with Arrows", flow_visualization)
cv2.waitKey(0)
cv2.destroyAllWindows()