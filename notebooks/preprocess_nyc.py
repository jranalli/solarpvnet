import os

import utils
from utils.fileio import files_of_type

from model.dataset_manipulation import limit_dataset_size

drive = "D:\\"
split_size = 625
n_subset = 1000
seed = 42

root = os.path.join(drive, "solardnn\\NYC")

img_dir = os.path.join(root, "img")
mask_dir = os.path.join(root, "mask")

tile_dir = os.path.join(root, "tiles\\img")
mask_tile_dir = os.path.join(root, "tiles\\mask")

subset_dir = os.path.join(root, "tile_subsets")

print("== Convert ZIP to PNG== ")
# Already done
# utils.zip_to_png(zip_file, img_dir)


print("== Generate Blank JSON ==")
utils.generate_blank_json_dir(img_dir)


print("== JSON to Mask ==")
fns = files_of_type(img_dir, "*.json")
for fn in fns:
    utils.labelme_json_to_binary(fn, mask_dir,
                                 label_name_to_value={"_background_": 0,
                                                      "maybe": 0,
                                                      "notpv": 0,
                                                      "pv": 255})


# print("== Slice Images ==")
# fns = files_of_type(img_dir, "*.png")
# for fn in fns:
#     utils.slice_image(fn, split_size, split_size, tile_dir)
#
#
# print("== Slice Masks ==")
# fns = files_of_type(mask_dir, "*.png")
# for fn in fns:
#     utils.slice_image(fn, split_size, split_size, mask_tile_dir)


print("== Delete Blanks ==")
utils.delete_blank_tiles(tile_dir, mask_tile_dir, maxfrac=0, seed=None)

print("== Make Subsets ==")
limit_dataset_size(tile_dir, mask_tile_dir, subset_dir, n_limit=n_subset,
                   seed=seed)
