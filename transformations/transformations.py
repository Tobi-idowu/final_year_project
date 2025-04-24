import numpy as np
import cv2
from PIL import Image
from cellpose import models
import tifffile as tiff


def distance_transform(file_path = "../data/segmented_data/segmentations/AB060922a_Job3_0240_seg.npy"):
    #read in the image
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

def optic_flow(image_path1 = "../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0006.tif", image_path2 = "data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0007.tif"):
    #read in the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

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

def cellpose_gradient_mask(image_path = "../data/segmented_data/images/pupa_1_stage_1_cropped/pupa_1_stage_1_cropped_0000.tif"):
    #read in the image
    image = tiff.imread(image_path)

    #instantiate the cellpose model
    model = models.Cellpose(model_type='cyto', gpu=True)

    #unpack cellpose's output
    _, flows, _, _ = model.eval([image], diameter=60, channels=[0, 0])

    #combine dx and dy for each pixel into a list of [dx, dy]
    gradient_vectors = np.stack((flows[0][1][0], flows[0][1][1]), axis=-1)

    #flatten the 3D array (height, width, 2) to a 2D array where each element is [dx, dy]
    gradient_vectors = gradient_vectors.reshape(-1,2)

    return gradient_vectors

def distance_transform_folder(folder_path, file_names):
    #array of distance transforms and segmentation maps
    dist_transforms = np.empty((len(file_names), 1024, 1024))
    binary_images = np.empty((len(file_names), 1024, 1024))

    for i in range(len(file_names)):
        #read in the segmentation
        file_path = folder_path + file_names[i][:-4] + "_seg.npy"
        try:
            data = np.load(file_path, allow_pickle=True).item()
        except:
            print(f"Error file: {file_path}")

        #access the image data from the dictionary
        image_data = data['outlines']

        #image_data[image_data > 0] = 1
        binary_image = (image_data <= 0).astype(np.uint8)

        #compute the distance transform
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)

        #store the current distance transform and segmentation map
        dist_transforms[i] = dist_transform
        binary_images[i] = binary_image

    return dist_transforms, binary_images

def optic_flow_folder(parent_path, folder_name, file_names):
    #these arrays will store the vectors decomposed as two directions 
    flows_dx = np.empty((len(file_names) - 1, 1024, 1024))
    flows_dy = np.empty((len(file_names) - 1, 1024, 1024))

    folder_path = parent_path + "/" + folder_name + "/"

    #set up the first iteration
    image2 = Image.open(folder_path + file_names[0])
    grey_image2 = np.array(image2.convert("L"))
    
    #loop through the frames in the video
    for i in range(len(file_names) - 1):
        grey_image1 = grey_image2

        image2 = Image.open(folder_path + file_names[i+1])
        grey_image2 = np.array(image2.convert("L"))

        #calculate optic flow
        flow = cv2.calcOpticalFlowFarneback(
            grey_image1, grey_image2, None, pyr_scale=0.5,
            levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )

        #extract and store the gradient information
        flows_dx[i] = flow[:, :, 0]
        flows_dy[i] = flow[:, :, 1]

    return flows_dx, flows_dy

def cellpose_gradient_mask_folder(model, parent_path, folder_name, file_names):
    folder_path = parent_path + "/" + folder_name + "/"

    #read in the frames of the video
    images = [tiff.imread(folder_path + f) for f in file_names]

    #unpack cellpose's output
    _, flows, _, _ = model.eval(images, diameter=60, channels=[0, 0])

    #these arrays will store the vectors decomposed as two directions
    gradient_masks_dx = np.empty((len(file_names)-1, 1024, 1024))
    gradient_masks_dy = np.empty((len(file_names)-1, 1024, 1024))

    #store the gradient masks
    for i in range(len(file_names)-1):
        gradient_masks_dx[i] = flows[i+1][1][0]
        gradient_masks_dy[i] = flows[i+1][1][1]

    return gradient_masks_dx, gradient_masks_dy

def main():
    print(distance_transform())
    print(optic_flow())
    print(cellpose_gradient_mask())

    return

if __name__ == "__main__":
    main()