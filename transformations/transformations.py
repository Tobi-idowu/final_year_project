import numpy as np
import cv2
from PIL import Image

from cellpose import models
import tifffile as tiff


def distance_transform(file_path = None):
    file_path = 'cellpose_all/pupa_1_stage_1_cropped_0000_seg.npy'

    data = np.load(file_path, allow_pickle=True)

    # Unpack the object
    data = data.item()

    # Access the image data from the dictionary (replace 'image_key' with the correct key)
    image_data = data['outlines']

    #image_data[image_data > 0] = 1
    binary_image = (image_data <= 0).astype(np.uint8)

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)  # L2 is Euclidean distance

    return dist_transform

def optic_flow(frame1 = None, frame2 = None):
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

    return flow

def cellpose_gradient_mask():
    image = tiff.imread(f'cellpose_all/pupa_1_stage_1_cropped_0001.tif')

    #this might have to be cyto3
    model = models.Cellpose(model_type='cyto', gpu=True)

    #unsure of the average diameter (average cell diameter in pixels)
    masks, flows, styles, diams = model.eval([image], diameter=60, channels=[0, 0])

    return flows[0]

def main():
    optic_flow()


main()