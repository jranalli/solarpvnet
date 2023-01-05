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
    for site in ["Germany", "Cal_Fresno", "Cal_Stockton", "France_ign", "France_google", "NYC", "combo_dataset"]:
        path = rf"F:\solardnn\{site}\tile_subsets\set0_seed42\train_mask_42"
        process_directory(path)

        path = rf"F:\solardnn\{site}\tile_subsets\set0_seed42\test_mask_42"
        process_directory(path)

    path = rf"F:\solardnn\Germany\tile_subsets\mask_set0_seed42"
    process_directory(path)
    path = rf"F:\solardnn\Germany\tile_subsets\mask_setfinal_seed42"
    process_directory(path)
    path = rf"F:\solardnn\Germany\tiles\mask"
    process_directory(path)