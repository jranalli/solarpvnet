import os

from split_image import split_image

import utils
from utils.fileio import files_of_type
from utils.configuration import get_loop_iter

drive = "d:"
split_size = 625
n_subset = 1000
seed = 42

root = os.path.join(drive, "datasets", "PV Aerial", "NY")

img_dir = os.path.join(root, "img")
mask_dir = os.path.join(root, "mask")

tile_dir = os.path.join(root, "tiles", "img")
mask_tile_dir = os.path.join(root, "tiles", "mask")

list_of_tiles_with_objects = os.path.join(root, "tiles", "object_tiles.txt")
list_of_tiles_blank = os.path.join(root, "tiles", "blank_tiles.txt")

#
# print("== Convert ZIP to PNG== ")
# # Already done
# # utils.zip_to_png(zip_file, img_dir)
#
#
# print("== Generate Blank JSON ==")
# utils.generate_blank_json_dir(img_dir)
#
#
# print("== JSON to Mask ==")
# fns = files_of_type(img_dir, "*.json")
# for fn in get_loop_iter(fns):
#     utils.labelme_json_to_binary(fn, mask_dir,
#                                  label_name_to_value={"_background_": 0,
#                                                       "maybe": 0,
#                                                       "notpv": 0,
#                                                       "pv": 255},
#                                  layer_order=["_background_", "pv", "maybe", "notpv"])
#
#
# print("== Slice Images ==")
# fns = files_of_type(img_dir, "*.png")
# n_row, n_col = utils.calc_rowcol(fns[0], split_size, split_size)
# for fn in get_loop_iter(fns):
#     split_image(fn, n_row, n_col, output_dir=tile_dir,
#                 should_square=False, should_cleanup=False, should_quiet=True)
#
#
# print("== Slice Masks ==")
# fns = files_of_type(mask_dir, "*.png")
# n_row, n_col = utils.calc_rowcol(fns[0], split_size, split_size)
# for fn in get_loop_iter(fns):
#     split_image(fn, n_row, n_col, output_dir=mask_tile_dir,
#                 should_square=False, should_cleanup=False, should_quiet=True)


print("== Generate Blank/NonBlank Lists ==")
utils.list_blank_tiles(tile_dir, mask_tile_dir, list_of_tiles_with_objects, list_of_tiles_blank)
