import os.path

from PIL import Image
import numpy as np

from utils.fileio import files_of_type


def delete_blank_tiles(img_dir, mask_dir, maxfrac=0, seed=None, img_ext="png"):
    """
    Look at a combined dataset of label masks and corresponding images. Find
    the masks that are blank and remove them from both parts of the dataset.

    Options allow for a fraction of the overall dataset to be blank.
    
    Parameters
    ----------
    img_dir: str
        Full path where the image files exist
    mask_dir: str
        Full path where the mask files exist (all filenames must repeat in 
        imgdir!)
    maxfrac: float (default 0)
        The fraction of the total dataset (between 0 and 1) that should be
        represented by blank files. Set to 0 to delete all blank files
    img_ext: str (default 'png')
        String of the image filetype
    seed: int (default None)
        A random seed that can be set to produce repeatable data. Caution! This
        will affect the global numpy.random.seed()!
    """

    # Get a list of all files in each location
    maskfns = files_of_type(mask_dir, "*." + img_ext)
    imgfns = files_of_type(mask_dir, "*." + img_ext)

    # Require that all the masks have a corresponding image file
    imgs = [os.path.basename(f) for f in imgfns]
    if not all(os.path.basename(fn) in imgs for fn in maskfns):
        raise ValueError("All mask files must have a matching image file.")

    # Idenfity blank masks
    blanks = []
    for fn in maskfns:
        with Image.open(fn) as img:
            if not img.getbbox():
                blanks.append(fn)

    # Calculate how many blanks we can keep
    nkeep = int(maxfrac * len(maskfns))

    if len(blanks) < nkeep:
        return  # Not enough blanks to hit the max, halt (keep all)
    else:
        # choose files to drop
        if seed is not None:
            np.random.seed(seed)
        keeps = np.random.choice(blanks, nkeep, replace=False)
        drops = list(filter(lambda x: x not in keeps, blanks))  # NOT keeps

        # Drop the file from both locations
        for f in drops:
            basefn = os.path.basename(f)

            os.remove(os.path.join(mask_dir, basefn))

        align_datasets(mask_dir, img_dir)


def list_blank_tiles(img_dir, mask_dir, obj_list_file, blank_list_file, img_ext="png", overwrite=True):
    """
    Look at a combined dataset of label masks and corresponding images. Find
    the masks that are blank and create separate lists of blank and not-blank
    tiles.

    Parameters
    ----------
    img_dir: str
        Full path where the image files exist
    mask_dir: str
        Full path where the mask files exist (all filenames must repeat in
        imgdir!)
    img_ext: str (default 'png')
        String of the image filetype
    overwrite: bool (default True)
        Overwrite the output files
    """

    if os.path.exists(obj_list_file) or os.path.exists(blank_list_file):
        if not overwrite:
            print("list_blank_tiles() - Output file exists, halting.")
            return

    # Create the outputs, which will hold basenames for each
    blanks = []
    have_objs = []

    # Get a list of all files in each location
    maskfns = files_of_type(mask_dir, "*." + img_ext)
    imgfns = files_of_type(img_dir, "*." + img_ext)

    # Require that all the masks have a corresponding image file
    imgs = [os.path.basename(f) for f in imgfns]
    masks = [os.path.basename(f) for f in maskfns]
    if not all(fn in imgs for fn in masks):
        raise ValueError("All mask files must have a matching image file.")

    # Look for potential images that aren't in the mask list, those are blank
    for fn in imgs:
        if fn not in masks:
            blanks.append(fn)

    # Loop over mask images and test if it's blank
    for fn in maskfns:
        with Image.open(fn) as img:
            # getbbox() returns bounding box of objects in the image
            if not img.getbbox():
                blanks.append(os.path.basename(fn))
            else:
                have_objs.append(os.path.basename(fn))

    # Make sure they're sorted
    blanks.sort()
    have_objs.sort()

    with open(obj_list_file, 'w') as f:
        f.write("\n".join([os.path.basename(fn) for fn in have_objs]))

    with open(blank_list_file, 'w') as f:
        f.write("\n".join([os.path.basename(fn) for fn in blanks]))


def align_datasets(folder_a, folder_b, img_ext="png"):
    """
    Take two folders, search through and delete any files that appear in only
    one. Caution! Files will be deleted!

    Parameters
    ----------
    folder_a: str
        Full path to first folder
    folder_b: str
        Full path to second folder
    img_ext: str (default "png")
        The extension of the filetype to look for
    """

    # Get a list of all files in each location
    fns_a = files_of_type(folder_a, "*." + img_ext)
    fns_b = files_of_type(folder_b, "*." + img_ext)

    bns_a = [os.path.basename(f) for f in fns_a]
    bns_b = [os.path.basename(f) for f in fns_b]

    # remove in each direction
    for fn in fns_a:
        if os.path.basename(fn) not in bns_b:
            os.remove(fn)

    for fn in fns_b:
        if os.path.basename(fn) not in bns_a:
            os.remove(fn)


# Sample run
png_tile_dir = "c:\\nycdata\\sample_subset\\test\\data"
mask_tile_dir = "c:\\nycdata\\sample_subset\\test\\masks"

if __name__ == '__main__':
    delete_blank_tiles(png_tile_dir, mask_tile_dir, seed=0)
