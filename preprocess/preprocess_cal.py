import os
from split_image import split_image
import utils
from utils.fileio import files_of_type
from model.dataset_manipulation import limit_dataset_size, reshape_and_save

drive = "D:\\"

roots = ["solardnn\\Cal_Oxnard\\",
         "solardnn\\Cal_Stockton\\",
         "solardnn\\Cal_Fresno\\"]
split_sizes = [500, 625, 625]

seed = 42
n_subset = 1000

model_size = 576

for root, split_size in zip(roots, split_sizes):
    root = os.path.join(drive, root)

    img_dir = os.path.join(root, "img")
    mask_dir = os.path.join(root, "mask")

    tile_dir = os.path.join(root, "tiles\\img")
    mask_tile_dir = os.path.join(root, "tiles\\mask")

    subset_dir = os.path.join(root, "tile_subsets")

    cal_json = os.path.join(drive, "solardnn\\Cal\\3385780_alt\\SolarArrayPolygons.json")
    cal_csv = os.path.join(drive, "solardnn\\Cal\\3385780_alt\\polygonDataExceptVertices.csv")

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

    print("== Delete Blanks ==")
    utils.delete_blank_tiles(tile_dir, mask_tile_dir, maxfrac=0, seed=None)
