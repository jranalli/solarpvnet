import os
import shutil

import numpy as np
from utils.generate_blank_json import generate_blank_json_dir
from utils.json_to_dataset import json_to_binary
from utils.slice_dataset_tiles import slice_image
from utils.fileio import files_of_type, verify_dir, is_dir_empty, clear_dir
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


def split_test_train(image_dir, mask_dir, output_root, test_ratio=0.1,
                     seed=None, overwrite=False, imext="png"):
    """
    Separate a test and training dataset.

    Parameters
    ----------
    image_dir: str
        Full path to directory containing the images
    mask_dir: str
        Full path to directory containing masks (must match filenames in
        image_dir.
    output_root: str
        Root path to directory where files will be output. Directory names
        coming out will be: test_img_SEED, test_mask_SEED, train_img_SEED,
        train_mask_SEED
    test_ratio: float (default 0.1)
        Fraction of images that should be made the test data
    seed: int (default None)
        Random seed for the random number generator. If None, no seed
        will be used. Caution! This will affect the global
        numpy.random.seed()!
    overwrite: bool (default False)
        Should files be overwritten in the target destinations?
    imext: str (default "png")
        The file extension of the images in the directory
    """
    # Make sure our output directories exist
    verify_dir(output_root)
    test_im_dir = os.path.join(output_root, f"test_img_{seed}")
    test_msk_dir = os.path.join(output_root, f"test_mask_{seed}")
    train_im_dir = os.path.join(output_root, f"train_img_{seed}")
    train_msk_dir = os.path.join(output_root, f"train_mask_{seed}")

    for idir in [test_im_dir, test_msk_dir, train_im_dir, train_msk_dir]:
        verify_dir(idir)
        if not is_dir_empty(idir):
            if not overwrite:
                print("Output path not empty. Skipping entire operation.")
                print(idir)
                return
            else:
                clear_dir(idir)

    all_fn = files_of_type(image_dir, "*." + imext)
    # Calculate how many blanks we can keep
    ntest = int(test_ratio * len(all_fn))

    # choose files to drop
    if seed is not None:
        np.random.seed(seed)
    test = list(np.random.choice(all_fn, ntest, replace=False))
    # train = list(filter(lambda x: x not in test, all_fn))  # NOT keeps
    # assert len(test) == ntest
    # assert len(test) + len(train) == len(all_fn)

    for f in all_fn:
        im_file = f
        msk_file = f.replace(image_dir, mask_dir)

        if im_file in test:
            out_im = test_im_dir
            out_msk = test_msk_dir
        else:
            out_im = train_im_dir
            out_msk = train_msk_dir

        shutil.copy(im_file, im_file.replace(image_dir, out_im))
        shutil.copy(msk_file, msk_file.replace(mask_dir, out_msk))