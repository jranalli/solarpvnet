import os
import utils

drive = "D:"
data_dir = os.path.join(drive, "datasets", "PV Aerial", "FR")


roots = [os.path.join(data_dir, "France_google"),
         os.path.join(data_dir, "France_ign")]


for root in roots:
    img_dir = os.path.join(root, "img")
    mask_dir = os.path.join(root, "mask")

    positive_tile_file = os.path.join(root, "positive_tiles.txt")
    negative_tile_file = os.path.join(root, "negative_tiles.txt")

    print("== Generate Blank Mask ==")
    utils.generate_blank_mask_dir(img_dir, mask_dir)

    print("== Generate Blank/NonBlank Lists ==")
    utils.list_blank_tiles(img_dir, mask_dir, positive_tile_file, negative_tile_file)
