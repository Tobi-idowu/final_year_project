from transformations.transformations import distance_transform_folder, optic_flow_folder, cellpose_gradient_mask_folder
from pathlib import Path
from cellpose import models
import numpy as np
import time
import h5py

# Start the timer
start_time = time.time()

def preprocess_data():
    parent_path = "data/segmented_data/images"
    parent_folder = Path(parent_path)

    #instantiate the cellpose model
    cellpose_model = models.Cellpose(model_type='cyto', gpu=True)

    #initialise the buffer variables
    training_examples = []
    targets = []
    buffer_len = 0
    buffer_num = 0

    folder_count = 0

    #initialise running totals
    running_sum = np.zeros((5, 1024, 1024))
    running_sum_sq = np.zeros((5, 1024, 1024))
    n = 0

    # empty the file
    with h5py.File("data/training_data.h5", "w") as f:
        pass

    #iterate through each folder
    for folder in parent_folder.iterdir():
        if folder.is_dir():
            folder_count += 1
            print(f"Folder {folder_count}: {folder.name}")

            #create array of all the names of the images in the folder
            files = [f.name for f in folder.iterdir() if f.is_file()]
            files.sort()

            #calculate distance transform for each image
            print(f"    Distance Transform")
            segmentations_path = parent_path[:-6] + "segmentations/"
            dist_transforms, binary_images = distance_transform_folder(segmentations_path, files)

            print_time_elapsed()

            #calculate the optical flow for each image
            print(f"    Optic Flow")
            optic_flows_dx, optic_flows_dy = optic_flow_folder(parent_path, folder.name, files)

            print_time_elapsed()

            #get the cellpose output for each image
            print(f"    Cellpose")
            gradient_masks_dx, gradient_masks_dy = cellpose_gradient_mask_folder(cellpose_model, parent_path, folder.name, files)

            print_time_elapsed()

            for i in range(len(files) - 1):
                # append the converted data to the training data
                training_example = [dist_transforms[i], optic_flows_dx[i], optic_flows_dy[i], gradient_masks_dx[i], gradient_masks_dy[i]]
                training_examples.append(training_example)
                target = binary_images[i+1]
                targets.append(target)

                print(f"    Example {i+1}/{len(files) - 1}")

                #update running totals
                running_sum += training_example
                running_sum_sq += np.array(training_example)**2
                n += 1024*1024
                buffer_len += 1

                #if the buffer has fills
                if buffer_len == 50:
                    # write buffer to file
                    with h5py.File("data/training_data.h5", "a") as f:
                        f.create_dataset(f"training_examples{buffer_num}", data = np.array(training_examples), compression = "gzip")
                        f.create_dataset(f"targets{buffer_num}", data = np.array(targets), compression = "gzip")

                    print(f"    Wrote buffer to file: {buffer_len} examples\n")

                    # reset buffer variables
                    training_examples = []
                    targets = []
                    buffer_len = 0
                    buffer_num += 1

        print_time_elapsed()

    # if buffer isnt empty write it to file
    if buffer_len != 0:
        with h5py.File("data/training_data.h5", "a") as f:
            f.create_dataset(f"training_examples{buffer_num}", data = np.array(training_examples), compression = "gzip")
            f.create_dataset(f"targets{buffer_num}", data = np.array(targets), compression = "gzip")

        print(f"Wrote buffer to file: {buffer_len} examples")

    print("==========CALCULATE MEAN AND SD==========")

    # compute the mean and standard deviation for each channel
    mean = running_sum.sum(axis=(1,2)) / n
    variance = (running_sum_sq.sum(axis=(1,2)) / n) - mean**2
    sd = np.sqrt(variance)

    print("==========STORE MEAN AND SD==========")

    #write the mean and standard deviation to the file
    with h5py.File("data/training_data.h5", "a") as f:
        f.create_dataset("mean", data = mean, compression = "gzip")
        f.create_dataset("sd", data = sd, compression = "gzip")

    print("==========MEAN AND SD STORED==========")

    print_time_elapsed()

    return


def normalise_data():
    print("==========NORMALISING DATA==========")

    buffer_num = 0
    key = "training_examples0"

    mean = None
    sd = None

    with h5py.File("data/training_data.h5", "r") as f:
        mean = f["mean"][:]
        sd = f["sd"][:]

    print(mean)
    print(sd)
    print()

    with h5py.File("data/training_data.h5", "r+") as f:
        #for each buffer in file
        while key in f:
            print(f"Buffer num: {buffer_num}")

            #normalise the data
            unnormalised_data = f[key][:]
            normalised_data = (unnormalised_data - mean[:, np.newaxis, np.newaxis]) / sd[:, np.newaxis, np.newaxis]

            #reshape the data for the neural network
            f.create_dataset(f"normalised_training_examples{buffer_num}", data = normalised_data.transpose(0, 2, 3, 1), compression = "gzip")

            #remove the unnormalised buffer
            del f[key]

            #set up next iteration
            buffer_num += 1
            key = f"training_examples{buffer_num}"

            print_time_elapsed()

    print(buffer_num)

    return


def print_time_elapsed():
    # End the timer
    end_time = time.time()

    # convert elapsed seconds to hours minutes and seconds
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600  # Calculate hours
    minutes = (elapsed_time % 3600) // 60  # Calculate minutes
    seconds = elapsed_time % 60  # Calculate remaining seconds

    print(f"    Elapsed time: {hours} hours, {minutes} minutes, {seconds} seconds\n")

    
if __name__ == "__main__":
    preprocess_data()
    normalise_data()
