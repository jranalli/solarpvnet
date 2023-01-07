from PIL import Image
import numpy as np
import glob
import shutil
import os

from utils.fileio import verify_dir, files_of_type

# https://github.com/whiplashoo/split-image
from split_image import split_image


def calc_rowcol(img_file, slice_width, slice_height):
    """
    Calculate the number of rows and columns that an image can be split into
    based on a specificed tile size.

    Parameters
    ----------
    img_file: str
        Full file path of the image
    slice_width: int
        slice width in pixels. Must result in integer number of slices
    slice_height: int
        slice height in pixels. Must result in integer number of slices

    Returns
    -------
    n_row, n_col: (int, int)
        The number of rows and columns that make up the image
    """
    with Image.open(img_file) as img:
        imwidth, imheight = img.size
        h_slices = int(np.ceil(imwidth / slice_width))
        v_slices = int(np.ceil(imheight / slice_height))
    return v_slices, h_slices


# Deprecated due to updates in split_image
# def slice_image(img_file, n_rows, n_cols, out_dir=None):
#     """
#     Slice an image into tiles of a given width and height. Overwrites existing
#     files.
#
#     Parameters
#     ----------
#     img_file: str
#         Full file path of the image
#     n_rows: int
#         Number of tile rows
#     n_cols: int
#         Number of tile columns
#     out_dir: str (default None)
#         Full output path for where to save the files. If None, don't move files
#     """
#
#     # Perform the split
#     split_image(img_file, n_rows, n_cols,
#                 should_square=False,
#                 should_cleanup=False,
#                 should_quiet=True,
#                 output_dir=out_dir)


# Example directory for testing
mydir = 'C:\\nycdata\\tst\\'
outdir = 'C:\\nycdata\\tstout\\'

if __name__ == "__main__":
    for fn in files_of_type(mydir, "*.png"):
        split_image(fn, 8, 8, False, False, True, output_dir=outdir)
