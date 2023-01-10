import PIL
import numpy as np

from utils.fileio import files_of_type


# Utility for normalizing all the images to the same scaling and color.

def process_directory(mydir):

    files = files_of_type(mydir, "*.png")

    for f in files:
        im = PIL.Image.open(f)
        im = np.asarray(im, dtype=np.float32)

        im = im/np.max(im) * 255

        lbl_pil = PIL.Image.fromarray(im.astype(np.uint8), mode="P")

        # Generate the output filename
        lbl_pil.save(f)


if __name__=="__main__":
    path = rf"d:\data\solardnn\DE-G\tiles\mask"
    process_directory(path)
