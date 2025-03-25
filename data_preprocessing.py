from transformations.transformations import distance_transform, optic_flow, cellpose_gradient_mask, distance_transform_folder, optic_flow_folder, cellpose_gradient_mask_folder
from pathlib import Path
from cellpose import models
import numpy as np
import time


def main():
    parent_path = "data/segmented_data/images"
    parent_folder = Path(parent_path)

    #instantiate the cellpose model
    #this might have to be cyto3
    cellpose_model = models.Cellpose(model_type='cyto', gpu=True)

    training_examples = []
    targets = []

    # Start the timer
    start_time = time.time()

    count = 0

    running_sum = np.zeros((5, 1024, 1024))
    running_sum_sq = np.zeros((5, 1024, 1024))
    n = 0

    for folder in parent_folder.iterdir():
        if folder.is_dir():
            count += 1
            print(f"Folder {count}: {folder.name}")

            files = [f.name for f in folder.iterdir() if f.is_file()]
            files.sort()

            #calculate distance transform for each image
            print(f"    Distance Transform")
            segmentations_path = parent_path[:-6] + "segmentations/"
            dist_transforms, binary_images = distance_transform_folder(segmentations_path, files)

            #calculate the optical flow for each image
            print(f"    Optic Flow")
            optic_flows_dx, optic_flows_dy = optic_flow_folder(parent_path, folder.name, files)

            #get the cellpose output for each image
            print(f"    Cellpose")
            gradient_masks_dx, gradient_masks_dy = cellpose_gradient_mask_folder(cellpose_model, parent_path, folder.name, files)

            for i in range(len(files) - 1):
                # append the converted data to the training data
                training_example = [dist_transforms[i], optic_flows_dx[i], optic_flows_dy[i], gradient_masks_dx[i], gradient_masks_dy[i]]
                training_examples.append(training_example)

                target = binary_images[i+1]
                targets.append(target)

                print(f"    Example {i+1}/{len(files) - 1}")

                running_sum += training_example
                running_sum_sq += np.array(training_example)**2
                n += 1024*1024

                # print("=======================================================")
                # print(dist_transforms[i])
                # print(distance_transform(segmentations_path + files[i][:-4] + "_seg.npy"))
                # print("=======================================================")

                # print("=======================================================")
                # print(optic_flows_dx[i], optic_flows_dy[i])
                # print(optic_flow(parent_path + "/" + folder.name + "/" + files[i], parent_path + "/" + folder.name + "/" + files[i+1]))
                # print("=======================================================")

                # print("=======================================================")
                # print(gradient_masks_dx[i], gradient_masks_dy[i])
                # print(cellpose_gradient_mask(parent_path + "/" + folder.name + "/" + files[i+1]))
                # print("=======================================================")

        # End the timer
        end_time = time.time()

        elapsed_time = end_time - start_time

        hours = elapsed_time // 3600  # Calculate hours
        minutes = (elapsed_time % 3600) // 60  # Calculate minutes
        seconds = elapsed_time % 60  # Calculate remaining seconds

        print(f"    Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")

    # convert training data into numpy arrays
    training_examples = np.array(training_examples)
    targets = np.array(targets)

    # compute the mean and standard deviation for each channel
    mean = running_sum.sum(dim=(1,2)) / n
    variance = (running_sum_sq.sum(dim=(1,2)) / n) - mean**2
    sd = np.sqrt(variance)

    # normalise the training examples
    normalised_examples = (training_examples - mean[:, np.newaxis, np.newaxis]) / sd[:, np.newaxis, np.newaxis]

    #save the training data to the numpy file
    np.savez_compressed("data/training_data.npz", normalised_examples = normalised_examples, targets = targets)

    # End the timer
    end_time = time.time()

    # convert elapsed seconds to hours minutes and seconds
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600  # Calculate hours
    minutes = (elapsed_time % 3600) // 60  # Calculate minutes
    seconds = elapsed_time % 60  # Calculate remaining seconds

    print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")

    return

    
if __name__ == "__main__":
    main()