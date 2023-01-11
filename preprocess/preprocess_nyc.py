import os

from split_image import split_image

import utils
from utils.fileio import files_of_type

from model.dataset_manipulation import test_train_valid_split

drive = "f:\\"
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
                                                      "pv": 255},
                                 layer_order=["_background_", "pv", "maybe", "notpv"])


print("== Slice Images ==")
fns = files_of_type(img_dir, "*.png")
n_row, n_col = utils.calc_rowcol(fns[0], split_size, split_size)
for fn in fns:
    split_image(fn, n_row, n_col, output_dir=tile_dir,
                should_square=False, should_cleanup=False, should_quiet=True)


print("== Slice Masks ==")
fns = files_of_type(mask_dir, "*.png")
n_row, n_col = utils.calc_rowcol(fns[0], split_size, split_size)
for fn in fns:
    split_image(fn, n_row, n_col, output_dir=mask_tile_dir,
                should_square=False, should_cleanup=False, should_quiet=True)


print("== Delete Blanks ==")
utils.delete_blank_tiles(tile_dir, mask_tile_dir, maxfrac=0, seed=None)
