import json
import os

from PIL import Image
import numpy as np
from utils.fileio import files_of_type


def generate_blank_json_file(json_file, shape):
    """
    Create a blank JSON file consistent with the structure of labelme outputs.

    Parameters
    ----------
    json_file: str
        Full path to the json file that should be created
    shape: tuple
        The (height, width) of the image
    """
    with open(json_file, "w") as file:
        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(json_file).replace(".json", ".png"),
            "imageData": None,
            "imageHeight": shape[1],
            "imageWidth": shape[0]
        }
        json_str = json.dumps(data, indent=2)
        file.write(json_str)


def generate_blank_json_dir(target_dir, json_dir=None, img_ext=".png"):
    """
    Loop over a directory and look for any files that don't have a
    corresponding JSON file. Create a blank JSON for any with missing JSON
    file.

    Parameters
    ----------
    target_dir: str
        full path of the directory to search
    json_dir: str or None (default None)
        full path of directory that contains the JSON files to compare with. If
        None, assumed to be the same as image dir.
    img_ext: str (default ".png")
        Image file extension to search for
    """
    if json_dir is None:
        json_dir = target_dir

    files = files_of_type(target_dir, "*" + img_ext)

    for f in files:
        jsonfn = os.path.join(json_dir, os.path.basename(f).replace(img_ext,
                                                                    ".json"))

        # Create the file if it doesn't exist
        if not os.path.exists(jsonfn):
            # Get the size of the existing image
            with Image.open(f) as im:
                shape = im.size
            generate_blank_json_file(jsonfn, shape)


def generate_blank_mask_dir(target_dir, mask_dir, img_ext=".png"):
    """
    Loop over a directory and look for any files that don't have a
    corresponding mask file. Create a blank mask for any with missing mask
    file.

    Parameters
    ----------
    target_dir: str
        full path of the directory to search
    mask_dir: str
        full path of directory that contains the mask files to compare with.
    img_ext: str (default ".png")
        Image file extension to search for
    """

    files = files_of_type(target_dir, "*" + img_ext)
    # masks = files_of_type(mask_dir, "*" + img_ext)

    for f in files:
        mask_fn = f.replace(target_dir, mask_dir)
        if not os.path.exists(mask_fn):
            # Create the file
            with Image.open(f) as im:
                shape = im.size
                arr = np.zeros(shape[:2], dtype=np.uint8)
                lbl_pil = Image.fromarray(arr, mode="P")
                # Generate the output filename
                lbl_pil.save(mask_fn)


# Example directory for testing
mydir = 'C:\\nycdata\\boro_queens_sp18_png\\'

if __name__ == "__main__":
    generate_blank_json_dir(mydir)
