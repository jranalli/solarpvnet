from utils.generate_blank_json import generate_blank_json_dir
from utils.json_to_dataset import json_to_binary
from utils.slice_dataset_tiles import slice_image
from utils.fileio import files_of_type
from utils.delete_blanks import delete_blank_tiles


def preprocess_files():
    mydir = "c:\\nycdata\\sample_dataset\\"
    mymaskdir = "c:\\nycdata\\sample_dataset_mask\\"

    tile_dir = "c:\\nycdata\\sample_dataset\\tiles"
    mask_tile_dir = "c:\\nycdata\\sample_dataset\\mask_tiles"

    generate_blank_json_dir(mydir)
    fns = files_of_type(mydir, "*.json")
    for fn in fns:
        json_to_binary(fn, mymaskdir, label_name_to_value={"_background_": 0, "maybe": 0, "notpv": 0, "pv": 255})

    fns = files_of_type(mydir, "*.png")
    for fn in fns:
        slice_image(fn, 625, 625, tile_dir)

    fns = files_of_type(mymaskdir, "*.png")
    for fn in fns:
        slice_image(fn, 625, 625, mask_tile_dir)

    delete_blank_tiles(tile_dir, mask_tile_dir, maxfrac=0, seed=None)


def preprocess_xy_images(image_list, mask_list, size=(576, 576)):
    """
    Read images, and resize, normalize and reshape for input to model.

    Parameters
    ----------
    image_list: list
        list of input image filenames
    mask_list: list
        list of mask image filenames
    size: tuple
        desired size of the image as tuple

    Returns
    -------
    x,y the output data with labels
    """
    import numpy as np
    from PIL import Image

    X_list = []
    Y_list = []
    for image, mask in zip(image_list, mask_list):
        im_X = Image.open(image)
        im_X = im_X.convert('RGB')
        im_X = im_X.resize(size)
        X_list.append(np.array(im_X))

        im_Y = Image.open(mask).resize(size)
        Y_list.append(np.array(im_Y))

    x = np.asarray(X_list)
    y = np.asarray(Y_list)

    # Normalize
    x = np.asarray(x, dtype=np.float32) / 255.0
    y = np.asarray(y, dtype=np.float32) / np.max(y)

    # Reshape X to (n_examples, size, size, 3[RGB])
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3)
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

    return x, y