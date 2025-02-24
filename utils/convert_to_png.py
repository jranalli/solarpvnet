from zipfile import ZipFile
import os
from PIL import Image
import numpy as np

from utils.fileio import verify_dir, is_dir_empty, clear_dir, files_of_type


def zip_to_png(source_zip, rgb_dir, ir_dir=None, img_ext="png",
               overwrite=False, verbose=True):
    """
    Process ZIP files from New York City GIS data download
         (https://gis.ny.gov/gateway/mg/2018/new_york_city/)

    Convert 4 channel JPEG2000 images into 3 channel RGB images and separate
    single channel images of the Infrared channel.

    Parameters
    ----------
    source_zip: str
        full path to the zip file downloaded from the database
    rgb_dir: str
        full path output directory for the rgb images
    ir_dir: str or None (default None)
        full path output directory for the alpha images. If None, alpha images
        are not saved.
    img_ext: str (default "png")
        Extension of the image file name to save as. Must be supported by PIL.
    overwrite: bool (default False)
        If output path(s) exist, should the operation be run anyway?
    verbose: bool (default True)
        Print a status update file by file?
    """

    # Make sure our output directories exist
    verify_dir(rgb_dir)
    if not is_dir_empty(rgb_dir):
        if not overwrite:
            print("RGB Output path not empty. Skipping entire operation.")
            return
        else:
            clear_dir(rgb_dir)

    if ir_dir is not None:
        verify_dir(ir_dir)
        if not is_dir_empty(ir_dir):
            if not overwrite:
                print("IR Output path not empty. Skipping entire operation.")
                return
            else:
                clear_dir(ir_dir)

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

                if verbose:
                    # Just output a status indicator
                    print(fn)

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
                    im.save(os.path.join(rgb_dir, fn_short + "." + img_ext))

                    # Layer 3 is Infrared
                    if ir_dir is not None:
                        alpha = Image.fromarray(a[:, :, -1].astype('uint8'))
                        alpha.save(os.path.join(ir_dir, fn_short +
                                                "." + img_ext))


def tif_to_png(tif_dir, overwrite=False, verbose=True, delete=False):
    """
    Convert a directory of tif images to pngs

    Parameters
    ----------
    tif_dir: str
        directory where tif images are stored
    overwrite: bool (default: False)
        Overwrite, or skip if exists
    verbose: bool (default: True)
        Print filenames while running
    delete: bool (default: False)
        Delete tif file when done?
    """
    fns = files_of_type(tif_dir, "*.tif")
    for fn in fns:
        if verbose:
            print(fn)
        with Image.open(fn) as f:
            pfn = fn.replace(".tif", ".png")
            if os.path.exists(pfn) and not overwrite:
                print(f"File exists, skipping: {pfn}")
                continue
            else:
                f.save(pfn)

        if delete:
            os.remove(fn)


# Example dirs for testing
sourcefile = r"E:\datasets\PV Aerial\NY\Raw\boro_queens_sp18.zip"
outdir = r"E:\datasets\PV Aerial\NY\img"
outdir_a = r"E:\datasets\PV Aerial\NY\img_ir"

if __name__ == "__main__":
    zip_to_png(sourcefile, outdir, outdir_a)
