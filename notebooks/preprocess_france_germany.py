import os
from model.dataset_manipulation import limit_dataset_size, reshape_and_save

drive = "D:\\"

roots = ["solardnn\\France_google\\",
         "solardnn\\France_ign\\",
         "solardnn\\Germany\\"]

n_subset = 1000
seed = 42

for root in roots:
    root = os.path.join(drive, root)

    tile_dir = os.path.join(root, "tiles\\img")
    mask_tile_dir = os.path.join(root, "tiles\\mask")

    subset_dir = os.path.join(root, "tile_subsets")

    print("== Make Subsets ==")
    limit_dataset_size(tile_dir, mask_tile_dir, subset_dir, n_limit=n_subset,
                       seed=seed)
