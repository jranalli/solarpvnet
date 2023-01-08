import os
from model.dataset_manipulation import make_combo_dataset_txt

sites = ["Germany", "Cal_Fresno", "Cal_Stockton", "France_ign", "France_google", "NYC"]

drive = "f:"
pathroot = os.path.join(drive, 'solardnn')

imgs = []
masks = []
img_roots = []
mask_roots = []

vimgs = []
vmasks = []
vimg_roots = []
vmask_roots = []
for site in sites:
    runroot = os.path.join(pathroot, site, "tiles")
    img_fn = os.path.join(runroot, "train_img_42.txt")
    msk_fn = os.path.join(runroot, "train_mask_42.txt")
    img_roots.append(os.path.join(runroot, "imgs"))
    mask_roots.append(os.path.join(runroot, "masks"))
    imgs.append(img_fn)
    masks.append(msk_fn)

    vimg_fn = os.path.join(runroot, "valid_img_42.txt")
    vmsk_fn = os.path.join(runroot, "valid_mask_42.txt")
    vimgs.append(img_fn)
    vmasks.append(msk_fn)

out_img = os.path.join(pathroot, "combo_dataset", "tiles", "train_img_42.txt")
out_msk = os.path.join(pathroot, "combo_dataset", "tiles", "train_mask_42.txt")

vout_img = os.path.join(pathroot, "combo_dataset", "tiles", "valid_img_42.txt")
vout_msk = os.path.join(pathroot, "combo_dataset", "tiles", "valid_mask_42.txt")


make_combo_dataset_txt(imgs, out_img, img_roots, total_imgs=720, seed=42)
make_combo_dataset_txt(masks, out_msk, mask_roots, total_imgs=720, seed=42)

make_combo_dataset_txt(vimgs, vout_img, img_roots, total_imgs=80, seed=42)
make_combo_dataset_txt(vmasks, vout_msk, mask_roots, total_imgs=80, seed=42)