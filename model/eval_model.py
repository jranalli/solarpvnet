import glob

from keras.optimizers import SGD

import csv

import segmentation_models as sm


import preprocess_sample
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np


def eval_model(input_dir, mask_dir, weight_file, result_file, pred_dir,
               plot_dir, backbone="resnet34", imsize=576):
    """

    Parameters
    ----------
    input_dir
    mask_dir
    weight_file
    result_file
    pred_dir
    plot_dir
    backbone
    imsize

    Returns
    -------

    """

    # test and create directories
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    else:
        shutil.rmtree(pred_dir)
        os.makedirs(pred_dir)

    if plot_dir is not None:
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        else:
            shutil.rmtree(plot_dir)
            os.makedirs(plot_dir)

    # Get the list of all input/output files
    images = glob.glob(os.path.join(input_dir, "*.png"))
    masks = glob.glob(os.path.join(mask_dir, "*.png"))

    # Load and reshape the data
    print("==== Load and Resize Data ====")
    x, y = preprocess_sample.preprocess_xy_images(images, masks,
                                                  (imsize, imsize))

    # Create the model and define metrics
    print("==== Create Model ====")
    model = sm.Unet(backbone,
                    encoder_weights='imagenet',
                    input_shape=(imsize, imsize, 3),
                    classes=1,
                    decoder_use_batchnorm=False)
    print("==== Compile Model ====")
    model.compile(
        optimizer=SGD(),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score,
                 sm.metrics.precision,
                 sm.metrics.recall,
                 sm.metrics.f1_score],
    )

    # Load the model meights
    print("==== Load Weights ====")
    model.load_weights(weight_file)

    # Perform the metric evaluation
    print("==== Perform Evaluation ====")
    res = model.evaluate(x, y, batch_size=16, verbose=1)

    # Write to file
    print("==== Save Evaluation ====")
    csv_cols = ["weight file"] + list(model.metrics_names)
    csv_row = [os.path.basename(weight_file)] + list(res)

    print(csv_cols)
    print(csv_row)

    with open(result_file, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_cols)
        writer.writerow(csv_row)

    # Perform the prediction (images)
    print("==== Perform Predictions ====")
    pred_imgs = model.predict(x, batch_size=16)
    pred_imgs = reshape_arr(pred_imgs)

    x = reshape_arr(x)
    y = reshape_arr(y)

    # Save plots and images
    print("==== Save Predictions ====")
    figsize = 7
    cols = 4
    alpha = 0.5

    for im_id, imname in enumerate(images):
        img_name = os.path.basename(imname)
        plt.imsave(os.path.join(pred_dir, img_name), pred_imgs[im_id],
                   cmap=plt.cm.gray)

        if plot_dir is not None:
            fig, axes = plt.subplots(1, cols,
                                     figsize=(cols * figsize, figsize))
            axes[0].set_title("original", fontsize=15)
            axes[1].set_title("ground truth", fontsize=15)
            axes[2].set_title("prediction", fontsize=15)
            axes[3].set_title("overlay", fontsize=15)
            axes[0].imshow(x[im_id], cmap=get_cmap(x))
            axes[0].set_axis_off()
            axes[1].imshow(y[im_id], cmap=get_cmap(y))
            axes[1].set_axis_off()

            axes[2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[2].set_axis_off()
            axes[3].imshow(x[im_id], cmap=get_cmap(x))
            axes[3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id],
                                                     desired_size=imsize)),
                           cmap=get_cmap(pred_imgs),
                           alpha=alpha)
            axes[3].set_axis_off()
            fig.savefig(os.path.join(plot_dir, img_name))


def zero_pad_mask(mask, desired_size):
    pad = (desired_size - mask.shape[0]) // 2
    padded_mask = np.pad(mask, pad, mode="constant")
    return padded_mask


def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'


def mask_to_red(mask):
    """
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    """
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size, img_size)
    c2 = np.zeros((img_size, img_size))
    c3 = np.zeros((img_size, img_size))
    c4 = mask.reshape(img_size, img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


if __name__ == '__main__':
    mysize = 576

    mybackbone = "resnet34"
    myseed = 42

    myimages = f"c:\\nycdata\\sample_subset\\tiles\\test_imgs_{myseed}"
    mymasks = f"c:\\nycdata\\sample_subset\\tiles\\test_imgs_{myseed}"
    myweightfile = f"c:\\nycdata\\sample_subset\\results\\{mybackbone}_{myseed}_weights_best.h5"

    mypreddir = f"c:\\nycdata\\sample_subset\\results\\results_{mybackbone}_{myseed}\\pred"
    myplotdir = f"c:\\nycdata\\sample_subset\\results\\results_{mybackbone}_{myseed}\\plot"
    myresultfile = "c:\\nycdata\\sample_subset\\results\\performance.csv"
    eval_model(myimages, mymasks, myweightfile, myresultfile, mypreddir,
               myplotdir, backbone=mybackbone, imsize=mysize)
