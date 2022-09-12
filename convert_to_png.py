from zipfile import ZipFile
import os
from PIL import Image
import numpy as np

sourcefile = "data/boro_queens_sp18.zip"
outdir = "data/boro_queens_sp18_png"
outdir_a = "data/boro_queens_sp18_alpha"


def verify_dir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)


def to_png(source_zip, rgb_dir, ir_dir, ir_suffix="_alpha"):

    # Open the zip file for internal access
    with ZipFile(source_zip, "r") as zipdat:

        # Get a list of all files within the zip
        fns = zipdat.namelist()

        # loop over each file within the zip
        for fn in fns:

            # Separate the file name and its extension for this file
            fn_short, ext = os.path.splitext(fn)

            # Check to see if it's an image
            if ext == ".jp2":

                # Just output a status indicator
                print(fn)

                # Make sure our output directories exist
                verify_dir(rgb_dir)
                verify_dir(ir_dir)

                # Get access to this individual picture file within the zip
                with zipdat.open(fn) as file:

                    # Read it in as a PIL object
                    pic = Image.open(file)

                    # Images have 4 layers (RGB + Infrared).
                    # Convert to NumPy Array for manipulation
                    x, y = pic.size
                    a = np.array(pic.getdata()).reshape(x, y, 4)

                    # Separate RGB and Infrared channels and save each into
                    # a PNG file
                    # Layers 0-2 are RGB
                    im = Image.fromarray(a[:, :, :-1].astype('uint8'))
                    im.save(os.path.join(rgb_dir, fn_short + ".png"))

                    # Layer 3 is Infrared
                    alpha = Image.fromarray(a[:, :, -1].astype('uint8'))
                    alpha.save(os.path.join(ir_dir,
                                            fn_short + ir_suffix + ".png"))


if __name__ == "__main__":
    to_png(sourcefile, outdir, outdir_a)
