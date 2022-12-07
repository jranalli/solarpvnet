import glob
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageFilter, ImageColor

from utils.fileio import verify_dir

import importlib.util


def model_boundary_plot(im_dir, truth_dir, pred_dirs, out_dir, dpi=300, verbose=True, model_names=["CA-F", "CA-S", "FR-I", "FR-G", "DE-G", "NY-Q"], colors=["#0000ff", "#000088", "#00ff00", "#008800", "#ff00ff", "#ff8800"]):
    img_paths = glob.glob(os.path.join(im_dir, "*.png"))
    imgs = [os.path.basename(img_path) for img_path in img_paths]
    titles = ['img', 'truth', 'predictions']

    figsize = 4
    cols = 3

    if importlib.util.find_spec("tqdm"):
        from tqdm import tqdm
        looper = tqdm(imgs)
    else:
        looper = imgs

    for i, img in enumerate(looper):
        if verbose and not importlib.util.find_spec("tqdm"):
            print(f"{i}/{len(imgs)}")

        fig, axes = plt.subplots(1, cols, figsize=(cols * figsize, figsize))

        for axis, title in zip(axes, titles):
            axis.set_axis_off()
            axis.set_title(title, fontsize=15)

        img_dat = Image.open(os.path.join(im_dir, img))
        axes[0].imshow(img_dat)
        truth_dat = Image.open(os.path.join(truth_dir, img))
        axes[1].imshow(truth_dat)

        handles, labels = axes[2].get_legend_handles_labels()

        # Plot image & truth border
        axes[2].imshow(img_dat, alpha=0.5)
        truth_brd = bkg_to_alpha(
            mask_colorize(
                mask_to_boundary(truth_dat, size=9),
                "#ff0000"))
        axes[2].imshow(truth_brd)

        legend_entry = Line2D([0], [0], label="Truth", color="#ff0000")
        handles.append(legend_entry)

        truth_dat.close()
        img_dat.close()

        # Plot prediction borders
        for pred_dir, model, color in zip(pred_dirs, model_names, colors):
            msk_im = Image.open(os.path.join(pred_dir, img))
            msk = msk_im.resize(img_dat.size)
            msk_im.close()

            msk = binarize(msk, 50)
            brd = bkg_to_alpha(
                mask_colorize(
                    mask_to_boundary(msk, size=3),
                    color=color))
            axes[2].imshow(brd)
            legend_entry = Line2D([0], [0], label=model, color=color)
            handles.append(legend_entry)
        plt.legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper left")
        verify_dir(out_dir)
        fig.savefig(os.path.join(out_dir, img), dpi=dpi)
        # plt.show()
        plt.close(fig)






def multimodel_plot(im_src, truth_dir, pred_dirs, out_dir, dpi=300, verbose=True, model_names=["CA-F", "CA-S", "FR-I", "FR-G", "DE-G", "NY-Q"]):

    bkg_alpha = 0.5
    fg_alpha = 0.5
    overlay_color = "#ff0000"

    img_paths = glob.glob(os.path.join(im_src, "*.png"))
    imgs = [os.path.basename(img_path) for img_path in img_paths]

    titles = ['img', 'truth']
    [titles.append(mdl) for mdl in model_names]

    figsize = 7
    cols = len(titles)

    if importlib.util.find_spec("tqdm"):
        from tqdm import tqdm
        looper = tqdm(imgs)
    else:
        looper = imgs

    for img in looper:
        if verbose and not importlib.util.find_spec("tqdm"):
            print(f"{i}/{len(imgs)}")

        fig, axes = plt.subplots(1, cols, figsize=(cols * figsize, figsize))
        for axis, title in zip(axes, titles):
            axis.set_axis_off()
            axis.set_title(title, fontsize=15)

        img_dat = Image.open(os.path.join(im_src, img))
        truth_dat = Image.open(os.path.join(truth_dir, img))

        axes[0].imshow(img_dat)
        axes[1].imshow(img_dat, alpha=bkg_alpha)
        axes[1].imshow(bkg_to_alpha(mask_colorize(truth_dat, overlay_color)), alpha=fg_alpha)

        for i, pred_dir in enumerate(pred_dirs):
            msk_im = Image.open(os.path.join(pred_dir, img))
            msk = msk_im.resize(img_dat.size)
            msk_im.close()

            # Binarize, colorize and remove bkg
            msk = binarize(msk, 50)
            msk = bkg_to_alpha(
                mask_colorize(msk, color=overlay_color))

            axes[i+2].imshow(img_dat, alpha=bkg_alpha)
            axes[i+2].imshow(msk, alpha=fg_alpha)

        verify_dir(out_dir)
        fig.savefig(os.path.join(out_dir, img), dpi=dpi)
        plt.close(fig)


def mask_to_boundary(mask, size=3, color=[255, 255, 255]):
    rgb = mask.convert(mode="RGB")
    dil = rgb.filter(ImageFilter.MaxFilter(size))
    ero = rgb.filter(ImageFilter.MinFilter(size))
    dif = np.asarray(dil) - np.asarray(ero)
    return Image.fromarray(dif)


def mask_colorize(mask, color=[255, 0, 0]):
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("RGB"))
    else:
        arr = mask

    if isinstance(color, str):
        color = ImageColor.getrgb(color)

    arr = (np.array(color) * arr / np.max(arr)).astype(np.uint8)
    if isinstance(mask, Image.Image):
        return Image.fromarray(arr)
    else:
        return arr

def bkg_to_alpha(mask, thresh=0, bkgalpha=0):
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask)
    else:
        arr = mask
    c1 = arr[:, :, 0]
    c2 = arr[:, :, 1]
    c3 = arr[:, :, 2]
    c4 = 255 * np.ones_like(arr[:, :, 0])
    c4[np.logical_not(np.any(arr > thresh, axis=-1))] = bkgalpha
    arr = np.stack((c1, c2, c3, c4), axis=-1).astype(np.uint8)
    if isinstance(mask, Image.Image):
        return Image.fromarray(arr)
    else:
        return arr

def binarize(image, thresh=0):
    image = image.point(lambda p: 255 if p > thresh else 0)
    return image


def demo_manual():
    fn = "002140_12.png"
    base = fr"d:\solardnn\tst\img\{fn}"
    baseim = Image.open(base)
    plt.imshow(baseim, alpha=0.75)

    truth = fr"d:\solardnn\tst\mask\{fn}"
    f = Image.open(truth)
    g = bkg_to_alpha(mask_colorize(mask_to_boundary(f, size=9), "#ff0000"))
    plt.imshow(g)

    handles, labels = plt.gca().get_legend_handles_labels()
    legend_entry = Line2D([0], [0], label="Truth", color="#ff0000")
    handles.append(legend_entry)

    for model, color in zip(["CA-F", "DE-G", "NY-Q"],
                            ["#00ff00", "#009900", "#005500"]):
        # for model, color in zip(["CA-F"],["#00ff00"]):
        im = fr"d:\solardnn\tst\{model}\{fn}"
        imo = Image.open(im)
        imo = imo.resize(baseim.size)
        imo = binarize(imo, 50)
        brd = bkg_to_alpha(
            mask_colorize(
                mask_to_boundary(imo, size=3),
                color=color))
        plt.imshow(brd)
        legend_entry = Line2D([0], [0], label=model, color=color)
        handles.append(legend_entry)

    plt.legend(handles=handles)
    plt.show()


if __name__ == "__main__":
    model_names = ["CA-F", "CA-S", "FR-I", "FR-G", "DE-G", "NY-Q"]

    for model in ["Cal_Fresno", "Cal_Stockton", "France_ign", "France_google", "Germany", "NYC"]:
        img_path = fr"d:\solardnn\{model}\tile_subsets\set0_seed42\test_img_42"
        truth_path = fr"d:\solardnn\{model}\tile_subsets\set0_seed42\test_mask_42"
        pred_dirs = [fr"D:\solardnn\results\results_set0_{model}_predbyCal_Fresno_resnet34_42\pred",
                     fr"D:\solardnn\results\results_set0_{model}_predbyCal_Stockton_resnet34_42\pred",
                     fr"D:\solardnn\results\results_set0_{model}_predbyFrance_ign_resnet34_42\pred",
                     fr"D:\solardnn\results\results_set0_{model}_predbyFrance_google_resnet34_42\pred",
                     fr"D:\solardnn\results\results_set0_{model}_predbyGermany_resnet34_42\pred",
                     fr"D:\solardnn\results\results_set0_{model}_predbyNYC_resnet34_42\pred"]
        outdir = fr"D:\solardnn\results\border_plots\{model}"

        print(f"\n=={model} BORDER==")
        model_boundary_plot(img_path, truth_path, pred_dirs, outdir, model_names=model_names)

        outdir = fr"D:\solardnn\results\combo_plots\{model}"

        print(f"\n=={model} COMBO==")
        multimodel_plot(img_path, truth_path, pred_dirs, outdir, model_names=model_names)