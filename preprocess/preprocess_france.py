import os
import utils
from model.dataset_manipulation import limit_dataset_size, reshape_and_save

drive = "D:"
data_dir = os.path.join(drive, "datasets", "PV Aerial", "FR")


roots = [os.path.join(data_dir, "France_google"),
         os.path.join(data_dir, "France_ign")]


for root in roots:
    img_dir = os.path.join(root, "img")
    mask_dir = os.path.join(root, "mask")

    list_of_tiles_with_objects = os.path.join(root, "object_tiles.txt")
    list_of_tiles_blank = os.path.join(root, "blank_tiles.txt")

    print("== Generate Blank Mask ==")
    utils.generate_blank_mask_dir(img_dir, mask_dir)

    print("== Generate Blank/NonBlank Lists ==")
    utils.list_blank_tiles(img_dir, mask_dir, list_of_tiles_with_objects, list_of_tiles_blank)
