import PIL
import numpy as np

from utils.fileio import files_of_type




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
    path = r"F:\solardnn\Germany\tile_subsets\set0_seed42\train_mask_42"

    process_directory(path)