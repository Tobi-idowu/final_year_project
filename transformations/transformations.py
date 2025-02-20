import numpy as np
import cv2
from PIL import Image

from cellpose import models
import tifffile as tiff

#import cupy as cp
#import cv2.cuda as cv2_cuda


def distance_transform(file_path = None):
    #read in the image
    file_path = '../data/segmented_data/segmentations/pupa_1_stage_1_cropped_0000_seg.npy'
    data = np.load(file_path, allow_pickle=True).item()

    #access the image data from the dictionary
    image_data = data['outlines']

    #image_data[image_data > 0] = 1
    binary_image = (image_data <= 0).astype(np.uint8)

    #compute the distance transform
    dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

    #flatten the distance transform into a 1d array
    dist_transform = dist_transform.reshape(-1, )

    return dist_transform

def optic_flow(image1 = None, image2 = None):
    #read in the images
    image1 = Image.open('../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0006.tif')
    image2 = Image.open('../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0007.tif')

    #convert them into greyscale
    grey_image1 = np.array(image1.convert("L"))
    grey_image2 = np.array(image2.convert("L"))

    #calculate optic flow
    flow = cv2.calcOpticalFlowFarneback(
        grey_image1, grey_image2, None, pyr_scale=0.5,
        levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )

    #flatten the 3D array (height, width, 2) to a 2D array where each element is [dx, dy]
    flow = flow.reshape(-1, 2)

    return flow

def cellpose_gradient_mask(image = None):
    #read in the image
    image = tiff.imread(f'../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0000.tif')

    #instantiate the cellpose model
    #this might have to be cyto3
    model = models.Cellpose(model_type='cyto', gpu=True)

    #unpack cellpose's output
    #unsure of the average diameter (average cell diameter in pixels)
    _, flows, _, _ = model.eval([image], diameter=60, channels=[0, 0])

    #combine dx and dy for each pixel into a list of [dx, dy]
    gradient_vectors = np.stack((flows[0][1][0], flows[0][1][1]), axis=-1)

    #flatten the 3D array (height, width, 2) to a 2D array where each element is [dx, dy]
    gradient_vectors = gradient_vectors.reshape(-1,2)

    return gradient_vectors

def main():
    #distance_transform()
    #print(optic_flow())
    cellpose_gradient_mask()


main()