from PIL import Image
import numpy as np
import glob
import shutil
import os

from utils.fileio import verify_dir, files_of_type

# https://github.com/whiplashoo/split-image
from split_image.split import split


def slice_image(image_file, slice_width, slice_height, out_dir=None):
    """
    Slice an image into tiles of a given width and height. Overwrites existing
    files.

    Parameters
    ----------
    image_file: str
        Full file path of the image
    slice_width: int
        slice width in pixels. Must result in integer number of slices
    slice_height: int
        slice height in pixels. Must result in integer number of slices
    out_dir: str (default None)
        Full output path for where to save the files. If None, don't move files
    """

    with Image.open(image_file) as img:
        imwidth, imheight = img.size
        h_slices = int(np.ceil(imwidth / slice_width))
        v_slices = int(np.ceil(imheight / slice_height))

        # Perform the split
        split(img, h_slices, v_slices, image_file, False)

        # move the files
        if out_dir is not None:
            verify_dir(out_dir)
            rootpath = os.path.dirname(image_file)
            basefn, ext = os.path.splitext(os.path.basename(image_file))

            fs = files_of_type(rootpath, basefn + "_*" + ext)
            for f in fs:
                thisfn = os.path.basename(f)
                shutil.move(f, os.path.join(out_dir, thisfn))




# Example directory for testing
mydir = 'C:\\nycdata\\tst\\'

if __name__ == "__main__":
    for fn in files_of_type(mydir, "*.png"):
        slice_image(fn, 625, 625)
