from PIL import Image
import numpy as np

# https://github.com/whiplashoo/split-image
from split_image.split import split

mydir = 'C:\\nycdata\\tst\\000137.png'

def slice_image(fn, slice_width, slice_height):
    with Image.open(fn) as img:
        imwidth, imheight = img.size
        h_slices = int(np.ceil(imwidth / slice_width))
        v_slices = int(np.ceil(imheight / slice_height))

        split(img, h_slices, v_slices, fn, False)



if __name__ == "__main__":
    slice_image(mydir, 1000, 1000)
