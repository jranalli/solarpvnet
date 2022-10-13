from PIL import Image
import numpy as np

from fileio import files_of_type

# https://github.com/whiplashoo/split-image
from split_image.split import split


def slice_image(image_file, slice_width, slice_height):
    """
    Slice an image into tiles of a given width and height

    Parameters
    ----------
    image_file: str
        Full file path of the image
    slice_width: int
        slice width in pixels. Must result in integer number of slices
    slice_height: int
        slice height in pixels. Must result in integer number of slices
    """
    with Image.open(image_file) as img:
        imwidth, imheight = img.size
        h_slices = int(np.ceil(imwidth / slice_width))
        v_slices = int(np.ceil(imheight / slice_height))

        split(img, h_slices, v_slices, image_file, False)


# Example directory for testing
mydir = 'C:\\nycdata\\tst\\'

if __name__ == "__main__":
    for fn in files_of_type(mydir, "*.png"):
        slice_image(fn, 625, 625)
