import os
import shutil

import numpy as np
from PIL import Image

import utils.fileio
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


def test_train_valid_split_list(img_files, mask_files, output_root, n_set=None,
                           test_train_valid=(0.2, 0.72, 0.08), seed=None, overwrite=False):
    """
    Split a list of image and mask filenames up into test_train_valid sets and save them
    to separate files.

    Parameters
    ----------
    img_files: list or tuple
        List of file basenames for images
    mask_files: list or tuple or None
        List of file basenames for masks.
    output_root: str
        Root path to directory where files will be output. Directory names
        coming out will be: test_img_SEED, test_mask_SEED, train_img_SEED,
        train_mask_SEED
    n_set: int (default 1000)
        Number of images in the total dataset. If None or 0, the whole list of
        images in the img_dir folder will be used.
    test_train_valid: list[float] (default [0.2, 0.72, 0.08])
        Fractions of test, train and valid sets relative to n_set.
    seed: int (default None)
        The seed to use to initialize np.random.seed. If None, ignore.
    overwrite: bool (default False)
        Should files be overwritten in the target destinations?

    Returns
    ----------
    Output files organized as a tuple as follows:
        (test_im_file, test_msk_file, train_im_file, train_msk_file, valid_im_file, valid_msk_file)
    """
    # Confirm that the lists are the same length
    if not len(mask_files) == len(img_files):
        raise ValueError("Image and Mask lists must be the same length.")

    # Make sure our output directories exist
    verify_dir(output_root)
    test_im_file = os.path.join(output_root, f"test_img_{seed}.txt")
    test_msk_file = os.path.join(output_root, f"test_mask_{seed}.txt")
    train_im_file = os.path.join(output_root, f"train_img_{seed}.txt")
    train_msk_file = os.path.join(output_root, f"train_mask_{seed}.txt")
    valid_im_file = os.path.join(output_root, f"valid_img_{seed}.txt")
    valid_msk_file = os.path.join(output_root, f"valid_mask_{seed}.txt")

    # Test for outputs already existing
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

    # Downselect a subset by choosing indices
    if n_set is None or n_set < 0:
        n_set = len(img_files)
    chosen_inds = list(np.random.choice(range(len(img_files)), n_set, replace=False))
    chosen_inds.sort()

    # Calculate how many belong in each split
    ntest = int(test_train_valid[0] * n_set)
    ntrain = int(test_train_valid[1] * n_set)
    nvalid = int(test_train_valid[2] * n_set)

    # Coerce to match n_set if they don't
    if ntest + ntrain + nvalid != n_set:
        print("Set does not split evenly, biasing towards train.")
        ntrain = n_set - ntest - nvalid

    # Choose indices to belong to each category.
    # Remove extras from the list.
    chosen_inds_cp = chosen_inds.copy()
    test = list(np.random.choice(chosen_inds_cp, ntest, replace=False))
    for item in test:
        chosen_inds_cp.remove(item)
    # Choose Train Files
    train = list(np.random.choice(chosen_inds_cp, ntrain, replace=False))
    for item in train:
        chosen_inds_cp.remove(item)
    # Valid remains
    assert len(chosen_inds_cp) == nvalid
    valid = chosen_inds_cp


    with open(test_im_file, "w") as test_im, \
            open(test_msk_file, "w") as test_msk, \
            open(train_im_file, "w") as train_im, \
            open(train_msk_file, "w") as train_msk, \
            open(valid_im_file, "w") as valid_im, \
            open(valid_msk_file, "w") as valid_msk:

        for ind in chosen_inds:
            im_file = img_files[ind]
            msk_file = mask_files[ind]

            # Store to file depending on name
            if ind in test:
                test_im.write(im_file+"\n")
                test_msk.write(msk_file+"\n")
            elif ind in train:
                train_im.write(im_file+"\n")
                train_msk.write(msk_file+"\n")
            else:  # it's in valid
                valid_im.write(im_file+"\n")
                valid_msk.write(msk_file+"\n")

    return test_im_file, test_msk_file, \
        train_im_file, train_msk_file, \
        valid_im_file, valid_msk_file


def test_train_valid_split(img_dir, mask_dir, output_root, n_set=None,
                           test_train_valid=(0.2, 0.72, 0.08),
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

    # Get full set of image files (basenames only)
    img_fn = files_of_type(img_dir, "*." + img_ext, fullpath=False)
    mask_fn = files_of_type(mask_dir, "*." + img_ext, fullpath=False)

    return test_train_valid_split_list(img_fn, mask_fn, output_root, n_set, test_train_valid, seed, overwrite)


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


def make_combo_dataset_txt(input_files, out_file, root_paths=None, weights=None, total_imgs=1000, seed=None, overwrite=False):
    """
    Make a combination dataset from other datasets by choosing a random sample.
    Builds a new text file. To work on both images and masks, this function
    could be run separately with the same seed on both the img and mask file
    lists.

    Parameters
    ----------
    input_files: list[str]
        A list of dataset root directories to build from
    out_file: str
        Full context of output file
    root_paths: list[str] or None
        List of root paths corresponding to each input file. These will be
        appended to each filename in the file. If None, nothing will be added.
    weights: iterable
        Scalable weights for each dataset to split by. Defaults to equal.
        Will be normalized to percentages.
    total_imgs: int
        size of dataset. Set to None to retain all images
    seed: int
        seed for random number generator
    overwrite: bool (default False)
        Should files be overwritten in the target destinations?
    """
    # Set seed
    if seed is not None:
        np.random.seed(seed)

    verify_dir(os.path.dirname(out_file))
    if os.path.exists(out_file) and not overwrite:
        print(f"Output file {out_file} exists. Skipping operation.")

    # Initialize root_paths if it doesn't exist
    if root_paths is None:
        root_paths = []
        for f in input_files:
            root_paths.append("")

    all_lists = []
    for fn, root in zip(input_files, root_paths):
        mylist = utils.fileio.read_file_list(fn, root)
        all_lists.append(mylist)

    if weights is None:
        weights = np.ones_like(input_files, dtype=np.float32)/len(input_files)

    im_each = np.floor(np.array(weights)/sum(weights) * total_imgs).astype("int")
    # Correct for rounding
    im_each[-1] = total_imgs - sum(im_each[:-1])

    biglist = []
    for set_fn, num in zip(all_lists, im_each):
        if total_imgs is None:
            subset = set_fn
        else:
            subset = list(np.random.choice(set_fn, num, replace=False))
        biglist += subset

    endl = "\n"
    with open(out_file, "w") as outf:
        for fn in biglist:
            # Concatenate subset together with endl in between and write
            outf.write(fn+endl)


def make_combo_dataset(data_paths, out_path, img_subpath="img", mask_subpath="mask", img_ext="png", weights=None, total_imgs=1000, seed=None):
    """
    Make a combination dataset from other datasets by choosing a random sample.
    Copies images from previous to new location.

    Parameters
    ----------
    data_paths: list
        A list of dataset root directories to build from
    out_path:
    img_subpath: str
        A subpath under data_path where images live
    mask_subpath: str
        A subpath under data_path where masks live
    img_ext: str
        The file extension for images
    weights: iterable
        Scalable weights for each dataset to split by. Defaults to equal.
        Will be normalized to percentages.
    total_imgs: int
        size of dataset. Set to None to retain all images
    seed: int
        seed for random number generator
    """
    # Set seed
    if seed is not None:
        np.random.seed(seed)

    img_path_out = os.path.join(out_path, img_subpath)
    mask_path_out = os.path.join(out_path, mask_subpath)
    verify_dir(img_path_out)
    verify_dir(mask_path_out)

    if not weights:
        weights = np.ones_like(data_paths, dtype=np.float32)/len(data_paths)

    im_each = np.floor(weights/sum(weights) * total_imgs).astype("int")
    # Correct for rounding
    im_each[-1] = total_imgs - sum(im_each[:-1])

    for path, num in zip(data_paths, im_each):
        img_path = os.path.join(path, img_subpath)
        mask_path = os.path.join(path, mask_subpath)
        all_fn = files_of_type(img_path, "*." + img_ext)
        if total_imgs is None:
            subset = all_fn
        else:
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