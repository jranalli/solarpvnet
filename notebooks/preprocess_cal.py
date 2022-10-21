import os
import utils
from utils.fileio import files_of_type

roots = ["D:\\solardnn\\Cal_Oxnard\\", "D:\\solardnn\\Cal_Stockton\\"]
split_sizes = [500]

for root, split_size in zip(roots, split_sizes):

    mydir = os.path.join(root, "img")
    mymaskdir = os.path.join(root, "mask")

    tile_dir = os.path.join(root, "tiles\\img")
    mask_tile_dir = os.path.join(root, "tiles\\mask")

    cal_json = "D:\\solardnn\\Cal\\3385780_alt\\SolarArrayPolygons.json"
    cal_csv = "D:\\solardnn\\Cal\\3385780_alt\\polygonDataExceptVertices.csv"


    print("== Convert TIF to PNG== ")
    utils.tif_to_png(mydir)

    print("== Generate LabelMe JSON ==")
    fns = files_of_type(mydir, "*.png")
    utils.json_to_dataset_cal(cal_json, cal_csv, fns)

    print("== Generate Blank JSON ==")
    utils.generate_blank_json_dir(mydir)

    print("== JSON to Mask ==")
    fns = files_of_type(mydir, "*.json")
    for fn in fns:
        utils.json_to_binary(fn, mymaskdir,
                       label_name_to_value={"_background_": 0, "maybe": 0,
                                            "notpv": 0, "pv": 255})

    print("== Slice Images ==")
    fns = files_of_type(mydir, "*.png")
    for fn in fns:
        utils.slice_image(fn, split_size, split_size, tile_dir)

    print("== Slice Masks ==")
    fns = files_of_type(mymaskdir, "*.png")
    for fn in fns:
        utils.slice_image(fn, split_size, split_size, mask_tile_dir)

    print("== Delete Blanks ==")
    utils.delete_blank_tiles(tile_dir, mask_tile_dir, maxfrac=0, seed=None)