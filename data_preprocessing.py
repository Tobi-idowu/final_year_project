from transformations.transformations import distance_transform, optic_flow, cellpose_gradient_mask, distance_transform_folder, optic_flow_folder, cellpose_gradient_mask_folder
from pathlib import Path
from cellpose import models


# print(distance_transform())
# print(optic_flow())
# print(cellpose_gradient_mask())

def main():
    parent_path = "data/segmented_data/images"
    parent_folder = Path(parent_path)

    #instantiate the cellpose model
    #this might have to be cyto3
    cellpose_model = models.Cellpose(model_type='cyto', gpu=True)

    for folder in parent_folder.iterdir():
        if folder.is_dir():
            files = [f.name for f in folder.iterdir() if f.is_file()]
            files.sort()

            # #calculate distance transform for each image
            # segmentations_path = parent_path[:-6] + "segmentations/"
            # dist_transforms = distance_transform_folder(segmentations_path, files)
            # print(dist_transforms)

            # #calculate the optical flow for each image 
            # optic_flows = optic_flow_folder(parent_path, folder.name, files)
            # print(optic_flows)

            #get the cellpose output for each image
            gradient_masks = cellpose_gradient_mask_folder(cellpose_model, parent_path, folder.name, files)
            print(gradient_masks)

        break 
    
if __name__ == "__main__":
    main()