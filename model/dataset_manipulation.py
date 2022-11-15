import os
import shutil

import numpy as np
from PIL import Image

from utils.fileio import files_of_type, verify_dir, is_dir_empty, clear_dir


def reshape_inputs(img_list, mask_list, size=(576, 576)):
    """
    Read images, and resize, normalize and reshape for input to model.

    Parameters
    ----------
    img_list: list
        list of input image filenames
    mask_list: list
        list of mask image filenames
    size: tuple
        desired size of the image as tuple

    Returns
    -------
    x,y the output data with labels
    """

    X_list = []
    Y_list = []
    for image, mask in zip(img_list, mask_list):
        im_X = Image.open(image)
        im_X = im_X.convert('RGB')
        im_X = im_X.resize(size)
        X_list.append(np.array(im_X))

        im_Y = Image.open(mask).resize(size)
        Y_list.append(np.array(im_Y))

    x = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(Y_list, dtype=np.float32)

    # Normalize
    x /= 255.0
    y /= np.max(y)

    # Reshape X to (n_examples, size, size, 3[RGB])
    x.shape = (x.shape[0], x.shape[1], x.shape[2], 3)
    y.shape = (y.shape[0], y.shape[1], y.shape[2], 1)
    #x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 3)
    #y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)

    return x, y


def reshape_and_save(img_list, out_dir, size=(576, 576)):
    """
    Read images, resize and save them

    Parameters
    ----------
    img_list: list
        list of input image filenames
    out_dir: str
        output directory
    size: tuple
        desired size of the image as tuple

    """

    for image, mask in img_list:
        fn_short = os.path.basename(image)
        im = Image.open(image)
        im = im.resize(size)
        im.save(os.path.join(out_dir, fn_short))


def split_test_train(img_dir, mask_dir, output_root, test_ratio=0.1, seed=None,
                     img_ext="png", overwrite=False):
    """
    Separate a test and training dataset.

    Parameters
    ----------
    img_dir: str
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
    img_ext: str (default "png")
        The file extension of the images in the directory

    Returns
    ----------
    Output directories organized as a tuple as follows:
        (train_im_dir, train_msk_dir, test_im_dir, test_msk_dir)
    """
    # Make sure our output directories exist
    verify_dir(output_root)
    test_im_dir = os.path.join(output_root, f"test_img_{seed}")
    test_msk_dir = os.path.join(output_root, f"test_mask_{seed}")
    train_im_dir = os.path.join(output_root, f"train_img_{seed}")
    train_msk_dir = os.path.join(output_root, f"train_mask_{seed}")

    for i_dir in [test_im_dir, test_msk_dir, train_im_dir, train_msk_dir]:
        verify_dir(i_dir)
        if not is_dir_empty(i_dir):
            if not overwrite:
                print("Output path not empty. Skipping entire operation.")
                print(i_dir)
                return
            else:
                clear_dir(i_dir)

    all_fn = files_of_type(img_dir, "*." + img_ext)
    # Calculate how many blanks we can keep
    ntest = int(test_ratio * len(all_fn))

    # choose files to mark as test
    if seed is not None:
        np.random.seed(seed)
    test = list(np.random.choice(all_fn, ntest, replace=False))

    # Save files
    for f in all_fn:
        im_file = f
        msk_file = f.replace(img_dir, mask_dir)

        # Generate filename depending on whether it is test or train
        if im_file in test:
            out_im = test_im_dir
            out_msk = test_msk_dir
        else:
            out_im = train_im_dir
            out_msk = train_msk_dir

        # Copy files to correct directory
        shutil.copy(im_file, im_file.replace(img_dir, out_im))
        shutil.copy(msk_file, msk_file.replace(mask_dir, out_msk))

    return train_im_dir, train_msk_dir, test_im_dir, test_msk_dir


def limit_dataset_size(img_dir, mask_dir, output_root, n_limit, seed,
                       img_ext="png", overwrite=False):
    """
    Split up a dataset into chunks of a fixed size.

    Parameters
    ----------
    img_dir: str
        Full path to directory containing the images
    mask_dir: str
        Full path to directory containing masks (must match filenames in
        image_dir.
    output_root: str
        Root path to directory where files will be output. Directory names
        coming out will be: img_setSET_seedSEED, mask_setSET_seedSEED, remainder
        will be marked setfinal
    n_limit: int
        Number of images for each subset
    seed: int (default None)
        Random seed for the random number generator. If None, no seed
        will be used. Caution! This will affect the global
        numpy.random.seed()!
    overwrite: bool (default False)
        Should files be overwritten in the target destinations?
    img_ext: str (default "png")
        The file extension of the images in the directory
    """

    # Set seed
    if seed is not None:
        np.random.seed(seed)

    # Init output dir and counter
    verify_dir(output_root)
    i = 0

    # Get all files
    all_fn = files_of_type(img_dir, "*." + img_ext)

    while len(all_fn) > n_limit:

        # choose files to mark as test
        subset = list(np.random.choice(all_fn, n_limit, replace=False))

        # get out dir names
        out_img_dir = os.path.join(output_root, f"img_set{i}_seed{seed}")
        out_msk_dir = os.path.join(output_root, f"mask_set{i}_seed{seed}")

        for i_dir in [out_img_dir, out_msk_dir]:
            verify_dir(i_dir)
            if not is_dir_empty(i_dir):
                if not overwrite:
                    print("Output path not empty. Skipping to next operation.")
                    print(i_dir)
                    continue
                else:
                    clear_dir(i_dir)

        # copy files
        for f in subset:
            all_fn.remove(f)
            im_file = f
            msk_file = f.replace(img_dir, mask_dir)
            shutil.copy(im_file, im_file.replace(img_dir, out_img_dir))
            shutil.copy(msk_file, msk_file.replace(mask_dir, out_msk_dir))

        i += 1

    # Copy the remainder
    out_img_dir = os.path.join(output_root, f"img_setfinal_seed{seed}")
    out_msk_dir = os.path.join(output_root, f"mask_setfinal_seed{seed}")

    for i_dir in [out_img_dir, out_msk_dir]:
        verify_dir(i_dir)
        if not is_dir_empty(i_dir):
            if not overwrite:
                print("Output path not empty. Skipping to next operation.")
                print(i_dir)
                continue
            else:
                clear_dir(i_dir)

    # copy files
    for f in all_fn:
        im_file = f
        msk_file = f.replace(img_dir, mask_dir)
        shutil.copy(im_file, im_file.replace(img_dir, out_img_dir))
        shutil.copy(msk_file, msk_file.replace(mask_dir, out_msk_dir))