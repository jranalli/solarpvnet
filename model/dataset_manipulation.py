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


def test_train_valid_split(img_dir, mask_dir, output_root, n_set=None,
                           exclude=None, test_train_valid=(0.2, 0.72, 0.08),
                           seed=None, img_ext="png", overwrite=False):
    """
    Split out subsets of the

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
    exclude: iterable[str] (default None)
        Exclude any filenames in the list. Routine will search for file
        basenames that appear in any way within exclude. Ignore if None.
    n_set: int (default 1000)
        Number of images in the total dataset. If None or 0, the whole list of
        images in the img_dir folder will be used.
    test_train_valid: list[float] (default [0.2, 0.72, 0.08])
        Fractions of test, train and valid sets relative to n_set.
    seed: int (default None)
        The seed to use to initialize np.random.seed. If None, ignore.
    img_ext: str (default "png")
        The file extension of the images in the directory
    overwrite: bool (default False)
        Should files be overwritten in the target destinations?

    Returns
    ----------
    Output files organized as a tuple as follows:
        (test_im_file, test_msk_file, train_im_file, train_msk_file, valid_im_file, valid_msk_file)
    """
    # Make sure our output directories exist
    verify_dir(output_root)
    test_im_file = os.path.join(output_root, f"test_img_{seed}.txt")
    test_msk_file = os.path.join(output_root, f"test_mask_{seed}.txt")
    train_im_file = os.path.join(output_root, f"train_img_{seed}.txt")
    train_msk_file = os.path.join(output_root, f"train_mask_{seed}.txt")
    valid_im_file = os.path.join(output_root, f"valid_img_{seed}.txt")
    valid_msk_file = os.path.join(output_root, f"valid_mask_{seed}.txt")

    #
    for i_file in [test_im_file, test_msk_file,
                   train_im_file, train_msk_file,
                   valid_im_file, valid_msk_file]:
        if os.path.exists(i_file):
            if not overwrite:
                print(f"Output file {i_file} exists. Skipping operation.")
                print(output_root)
                return test_im_file, test_msk_file, \
                    train_im_file, train_msk_file, \
                    valid_im_file, valid_msk_file
            else:  # Overwrite
                os.remove(i_file)

    # Set seed if exists
    if seed is not None:
        np.random.seed(seed)

    # Get full set of image files
    all_fn = files_of_type(img_dir, "*." + img_ext, fullpath=False)

    # If exclude provided, pop them out of all_fn
    if exclude is not None:
        for fn in all_fn.copy():
            # this handles full context pathnames in both. If speed is an issue
            # refactor to require basenames only in exclude
            if any(os.path.basename(fn) in fstr for fstr in exclude):
                all_fn.remove(fn)

    # Get the subset if requested to do so
    if n_set is not None and n_set > 0:
        all_fn = list(np.random.choice(all_fn, n_set, replace=False))
    else:
        n_set = len(all_fn)

    # Calculate how many belong in each split
    ntest = int(test_train_valid[0] * n_set)
    ntrain = int(test_train_valid[1] * n_set)
    nvalid = int(test_train_valid[2] * n_set)

    # Coerce to match n_split
    if ntest + ntrain + nvalid != n_set:
        print("Set does not split evenly, biasing towards train.")
        ntrain = n_set - ntest - nvalid

    # choose files to mark as test
    all_fn_cp = all_fn.copy()
    test = list(np.random.choice(all_fn_cp, ntest, replace=False))
    for item in test:
        all_fn_cp.remove(item)
    # Choose Train Files
    train = list(np.random.choice(all_fn_cp, ntrain, replace=False))
    for item in train:
        all_fn_cp.remove(item)
    # Valid remains
    assert len(all_fn_cp) == nvalid
    valid = all_fn_cp

    # Get file extensions for each images and masks
    img_extn = "." + img_ext  # this was easy
    # Find the filename of the first test image excluding extension
    first_im = os.path.splitext(os.path.basename(test[0]))[0]
    # Get the corresponding mask file
    mask_files = files_of_type(mask_dir, first_im + "*")
    mask_extn = os.path.splitext(mask_files[0])[-1]

    with open(test_im_file, "w") as test_im, \
            open(test_msk_file, "w") as test_msk, \
            open(train_im_file, "w") as train_im, \
            open(train_msk_file, "w") as train_msk, \
            open(valid_im_file, "w") as valid_im, \
            open(valid_msk_file, "w") as valid_msk:
        # Save files
        for f in all_fn:
            im_file = f
            msk_file = f.replace(img_dir, mask_dir).replace(img_extn, mask_extn)

            # Store to file depending on name
            if im_file in test:
                test_im.write(im_file+"\n")
                test_msk.write(msk_file+"\n")
            elif im_file in train:
                train_im.write(im_file+"\n")
                train_msk.write(msk_file+"\n")
            else:  # it's in valid
                valid_im.write(im_file+"\n")
                valid_msk.write(msk_file+"\n")

    return test_im_file, test_msk_file, \
        train_im_file, train_msk_file, \
        valid_im_file, valid_msk_file


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

def make_combo_dataset(data_paths, out_path, img_subpath="img", mask_subpath="mask", img_ext="png", weights=None, total_imgs=1000, seed=None):

    # Set seed
    if seed is not None:
        np.random.seed(seed)


    img_path_out = os.path.join(out_path, img_subpath)
    mask_path_out = os.path.join(out_path, mask_subpath)
    verify_dir(img_path_out)
    verify_dir(mask_path_out)

    if not weights:
        weights = np.ones_like(data_paths, dtype=np.float32)/len(data_paths)

    im_each = np.floor(weights * total_imgs).astype("int")
    # Correct for rounding
    im_each[-1] = total_imgs - sum(im_each[:-1])

    for path, num in zip(data_paths, im_each):
        img_path = os.path.join(path, img_subpath)
        mask_path = os.path.join(path, mask_subpath)
        all_fn = files_of_type(img_path, "*." + img_ext)
        subset = list(np.random.choice(all_fn, num, replace=False))

        for fn in subset:
            im_file = fn
            msk_file = fn.replace(img_subpath, mask_subpath)
            shutil.copy(im_file, im_file.replace(path, out_path))
            shutil.copy(msk_file, msk_file.replace(path, out_path))

if __name__ == "__main__":
    paths = [r"F:\solardnn\Cal_Fresno\tile_subsets\set0_seed42",
             r"F:\solardnn\Cal_Stockton\tile_subsets\set0_seed42",
             r"F:\solardnn\France_ign\tile_subsets\set0_seed42",
             r"F:\solardnn\France_google\tile_subsets\set0_seed42",
             r"F:\solardnn\Germany\tile_subsets\set0_seed42",
             r"F:\solardnn\NYC\tile_subsets\set0_seed42"
             ]
    img_path = "train_img_42"
    mask_path = "train_mask_42"
    out_path=r"F:\solardnn\combo_dataset"

    make_combo_dataset(paths, out_path, img_path, mask_path, total_imgs=800, seed=42)