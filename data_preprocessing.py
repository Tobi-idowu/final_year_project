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

    training_data = []

    # Start the timer
    start_time = time.time()

    count = 0

    for folder in parent_folder.iterdir():
        if folder.is_dir():
            count += 1
            print(f"Folder {count}: {folder.name}")

            files = [f.name for f in folder.iterdir() if f.is_file()]
            files.sort()

            #calculate distance transform for each image
            segmentations_path = parent_path[:-6] + "segmentations/"
            dist_transforms = distance_transform_folder(segmentations_path, files)

            #calculate the optical flow for each image
            optic_flows_dx, optic_flows_dy = optic_flow_folder(parent_path, folder.name, files)

            #get the cellpose output for each image
            gradient_masks_dx, gradient_masks_dy = cellpose_gradient_mask_folder(cellpose_model, parent_path, folder.name, files)

            for i in range(len(files) - 1):
                training_example = [dist_transforms[i], optic_flows_dx[i], optic_flows_dy[i], gradient_masks_dx[i], gradient_masks_dy[i], dist_transforms[i+1]]
                training_data.append(training_example)

                print(f"    File {i+1}/{len(files) - 1}  ")

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

    training_data = np.array(training_data)

    np.savez_compressed("data/training_data.npz", training_data = training_data)

    # End the timer
    end_time = time.time()

    elapsed_time = end_time - start_time

    hours = elapsed_time // 3600  # Calculate hours
    minutes = (elapsed_time % 3600) // 60  # Calculate minutes
    seconds = elapsed_time % 60  # Calculate remaining seconds

    print(f"Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")

    return

    
if __name__ == "__main__":
    main()