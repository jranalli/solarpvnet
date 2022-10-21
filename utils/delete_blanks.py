import os.path

from PIL import Image
import numpy as np

from utils.fileio import files_of_type


def delete_blank_tiles(imgdir, maskdir, maxfrac=0,
                       imgtype="png", seed=None):
    """
    Look at a combined dataset of label masks and corresponding images. Find
    the masks that are blank and remove them from both parts of the dataset.

    Options allow for a fraction of the overall dataset to be blank.
    
    Parameters
    ----------
    imgdir: str
        Full path where the image files exist
    maskdir: str
        Full path where the mask files exist (all filenames must repeat in 
        imgdir!)
    maxfrac: float (default 0)
        The fraction of the total dataset (between 0 and 1) that should be
        represented by blank files. Set to 0 to delete all blank files
    imgtype: str (default 'png')
        String of the image filetype
    seed: int (default None)
        A random seed that can be set to produce repeatable data. Caution! This
        will affect the global numpy.random.seed()!
    """

    # Get a list of all files in each location
    maskfns = files_of_type(maskdir, "*." + imgtype)
    imgfns = files_of_type(maskdir, "*." + imgtype)

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

            os.remove(os.path.join(maskdir, basefn))

        align_datasets(maskdir, imgdir)


def align_datasets(folder_a, folder_b, imgtype="png"):
    """
    Take two folders, search through and delete any files that appear in only
    one. Caution! Files will be deleted!

    Parameters
    ----------
    folder_a: str
        Full path to first folder
    folder_b: str
        Full path to second folder
    imgtype: str (default "png")
        The extension of the filetype to look for
    """

    # Get a list of all files in each location
    fns_a = files_of_type(folder_a, "*." + imgtype)
    fns_b = files_of_type(folder_b, "*." + imgtype)

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
