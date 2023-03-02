import os
from split_image import split_image
import utils
from utils.fileio import files_of_type

drive = "D:"
data_dir = os.path.join(drive, "datasets", "PV Aerial", "CA")
roots = [os.path.join(data_dir, "3385807_Oxnard"),
         os.path.join(data_dir, "3385804_Stockton"),
         os.path.join(data_dir, "3385828_Fresno")]
split_sizes = [500, 625, 625]

seed = 42
n_subset = 1000

model_size = 576

for root, split_size in zip(roots, split_sizes):

    img_dir = os.path.join(root)
    mask_dir = os.path.join(root, "mask")

    tile_dir = os.path.join(root, "tiles", "img")
    mask_tile_dir = os.path.join(root, "tiles", "mask")

    positive_tile_file = os.path.join(root, "tiles", "positive_tiles.txt")
    negative_tile_file = os.path.join(root, "tiles", "negative_tiles.txt")

    cal_json = os.path.join(data_dir, "3385780_alt", "SolarArrayPolygons.json")
    cal_csv = os.path.join(data_dir, "3385780_alt", "polygonDataExceptVertices.csv")

    print("== Convert TIF to PNG== ")
    utils.tif_to_png(img_dir, delete=True)

    print("== Generate LabelMe JSON ==")
    fns = files_of_type(img_dir, "*.png")
    utils.cal_to_labelme(fns, cal_json, cal_csv)

    print("== Generate Blank JSON ==")
    utils.generate_blank_json_dir(img_dir)

    print("== JSON to Mask ==")
    fns = files_of_type(img_dir, "*.json")
    for fn in fns:
        print(fn)
        utils.labelme_json_to_binary(fn, mask_dir,
                                     label_name_to_value={"_background_": 0,
                                                          "maybe": 0,
                                                          "notpv": 0,
                                                          "pv": 255})

    print("== Slice Images ==")
    fns = files_of_type(img_dir, "*.png")
    n_row, n_col = utils.calc_rowcol(fns[0], split_size, split_size)
    for fn in fns:
        split_image(fn, n_row, n_col, output_dir=tile_dir,
                    should_square=False, should_cleanup=False,
                    should_quiet=True)

    print("== Slice Masks ==")
    fns = files_of_type(mask_dir, "*.png")
    n_row, n_col = utils.calc_rowcol(fns[0], split_size, split_size)
    for fn in fns:
        split_image(fn, n_row, n_col, output_dir=mask_tile_dir,
                    should_square=False, should_cleanup=False,
                    should_quiet=True)

    print("== Generate List of Blanks ==")
    utils.list_blank_tiles(tile_dir, mask_tile_dir, positive_tile_file, negative_tile_file)
