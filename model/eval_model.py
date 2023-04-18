import glob

import csv

from model.dataset_manipulation import reshape_inputs
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from keras.optimizers import Adam, SGD
from sklearn.metrics import precision_recall_curve

import segmentation_models as sm

from utils.fileio import read_file_list, verify_dir, is_dir_empty
from utils.configuration import get_loop_iter


def eval_model(test_img_dir, test_mask_dir, test_img_file, test_mask_file, weight_file, result_file, pred_dir,
               plot_dir=None, backbone="resnet34", img_size=(576, 576), batchnorm=False, overwrite=False):
    """
    Perform the evaluation of the model

    Parameters
    ----------
    test_img_dir: str
        Directory with test images to predict. Use None for test_img_file that contains full paths.
    test_mask_dir: str
        Directory with test masks. Use None for test_img_file that contains full paths.
    test_img_file: str
        Full context of file containing list of train images
    test_mask_file: str
        Full context of file containing list of train mask images
    weight_file: str
        Full location of saved weights
    result_file: str
        Full location of file to save results to
    pred_dir: str
        Full location of path to save prediction images
    plot_dir: str
        Full location of path to save plot images
    backbone: str
        Model backbone
    img_size: tuple
        Image size in (xxx, yyy)
    batchnorm: bool (default: False)
        Use batchnorm
    overwrite: bool (default: False)
        Should files be overwritten?
    """

    # test and create directories
    verify_dir(pred_dir)
    if not is_dir_empty(pred_dir):
        if not overwrite:
            print("Prediction directory is not empty, skipping operation...")
            return
        else:
            shutil.rmtree(pred_dir)
            verify_dir(pred_dir)

    if plot_dir is not None:
        verify_dir(plot_dir)
        if not is_dir_empty(plot_dir):
            if not overwrite:
                print("Plot directory is not empty, skipping operation...")
                return
            else:
                shutil.rmtree(plot_dir)
                verify_dir(plot_dir)

    # Get the list of all input/output files
    images = read_file_list(test_img_file, test_img_dir)
    masks = read_file_list(test_mask_file, test_mask_dir)

    # Load and reshape the data
    print("==== Load and Resize Data ====")
    x, y = reshape_inputs(images, masks, img_size)

    # Create the model and define metrics
    print("==== Create Model ====")
    model = sm.Unet(backbone,
                    encoder_weights='imagenet',
                    input_shape=(img_size[0], img_size[1], 3),
                    classes=1,
                    decoder_use_batchnorm=batchnorm)
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

    for im_id, imname in enumerate(images):
        img_name = os.path.join(pred_dir, os.path.basename(imname))
        Image.fromarray((255*pred_imgs[im_id]).astype(np.uint8), mode="L").save(img_name)


def pr_curve(truth_dir, truth_file, pred_dir, result_file, overwrite=False):
    """
    Generate a precision-recall curve for a given model

    Parameters
    ----------
    truth_dir: str
        Directory with truth mask images. Use None for test_img_file that contains full paths.
    truth_file: str
        Full context of file containing list of truth mask images
    pred_dir: str
        Full location of path to prediction probability images
    result_file: str
        Full location of file to save results to
    overwrite: bool (default: False)
        Should files be overwritten?
    """

    # Check if file exists
    verify_dir(os.path.dirname(result_file))
    if os.path.isfile(result_file):
        if not overwrite:
            print("Result file exists, skipping operation...")
            return
        else:
            os.remove(result_file)

    # Get the list of all input/output files
    truth_files = read_file_list(truth_file, truth_dir)
    pred_files = [name.replace(os.path.dirname(name), pred_dir) for name in truth_files]

    # Load and reshape the data
    X_list = []
    Y_list = []
    for image, mask in get_loop_iter(zip(truth_files, pred_files)):
        im_Y = Image.open(mask).convert("L")  # Convert to grayscale
        Y_list.append(np.array(im_Y))
        size = im_Y.size
        im_Y.close()

        im_X = Image.open(image).convert("L")  # Convert to grayscale
        im_X = im_X.resize(size)
        X_list.append(np.array(im_X))
        im_X.close()

    x = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(Y_list, dtype=np.float32)

    # Normalize
    x /= np.max(x)
    y /= np.max(y)

    # Compute the curves, binarizing x
    p, r, _ = precision_recall_curve(x.flatten() > 0, y.flatten())

    # Write to a file
    df = pd.DataFrame({"precision": p, "recall": r})
    df.to_csv(result_file, index=False)

    # # plot the precision-recall curves
    # plt.plot(r, p, marker='.')
    # # axis labels
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # # show the legend
    # plt.legend()
    # # show the plot
    # plt.show()


def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


if __name__ == '__main__':
    pass
    # this is obsolete. Rewrite?
