import os
import utils

drive = "D:"
data_dir = os.path.join(drive, "datasets", "PV Aerial", "DE")

root = os.path.join(data_dir)

img_dir = os.path.join(root, "images")
mask_dir = os.path.join(root, "mask")
json_dir = os.path.join(root, "labels")

positive_tile_file = os.path.join(root, "positive_tiles.txt")
negative_tile_file = os.path.join(root, "negative_tiles.txt")

print("== Generate Masks from JSON ==")
fns = utils.fileio.files_of_type(json_dir, "*.json")
for fn in utils.configuration.get_loop_iter(fns):
    utils.labelme_json_to_binary(fn, mask_dir,
                                 label_name_to_value={"_background_": 0,
                                                      "maybe": 0,
                                                      "notpv": 0,
                                                      "PV": 255},
                                 layer_order=["_background_", "pv", "maybe", "notpv"])

print("== Generate Blank Masks ==")
utils.generate_blank_mask_dir(img_dir, mask_dir)

print("== Generate Blank/NonBlank Lists ==")
utils.list_blank_tiles(img_dir, mask_dir, positive_tile_file, negative_tile_file)
