import glob
# import argparse
#
# import tensorflow as tf
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam, SGD



import segmentation_models as sm


import preprocess_sample
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

MODEL_IMAGE_SIZE = 576

backbone = "resnet34"
inputpath = "c:\\nycdata\\sample_dataset_tiles\\"
maskpath = "c:\\nycdata\\sample_dataset_mask_tiles\\"
weightfile = "c:\\nycdata\\old_build_weights.h5"

result_dir = "c:\\nycdata\\model_out_old\\"
plot_dir = "c:\\nycdata\\plot_out_old\\"

def main():


    size = MODEL_IMAGE_SIZE

    # test and create directories
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    else:
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)

    if plot_dir is not None:
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)
        else:
            shutil.rmtree(plot_dir)
            os.makedirs(plot_dir)

    # Get the list of all input/output files
    orgs = glob.glob(inputpath + "*.png")
    masks = glob.glob(maskpath + "*.png")


    # Load and split the data
    x, y = preprocess_sample.preprocess_xy_images(orgs, masks, (size, size))

    # Create the model and define metrics
    model = sm.Unet(backbone,
                    encoder_weights='imagenet',
                    input_shape=(size, size, 3),
                    classes=1,
                    decoder_use_batchnorm=False)
    model.compile(
        optimizer=SGD(),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall],
    )

    # Load the model and make predictions
    model_filename = weightfile
    model.load_weights(model_filename)

    pred_imgs = model.predict(x, batch_size=16)
    pred_imgs = reshape_arr(pred_imgs)

    x = reshape_arr(x)
    y = reshape_arr(y)

    figsize = 7
    cols = 4
    alpha = 0.5

    for im_id, imname in enumerate(orgs):
        img_name = os.path.basename(imname)
        plt.imsave(os.path.join(result_dir, img_name), pred_imgs[im_id], cmap=plt.cm.gray)

        if plot_dir is not None:
            fig, axes = plt.subplots(1, cols, figsize=(cols * figsize, figsize))
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
            axes[3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=size)),
                           cmap=get_cmap(pred_imgs), alpha=alpha)
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
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size,img_size)
    c2 = np.zeros((img_size,img_size))
    c3 = np.zeros((img_size,img_size))
    c4 = mask.reshape(img_size,img_size)
    return np.stack((c1, c2, c3, c4), axis=-1)


if __name__ == '__main__':
    main()