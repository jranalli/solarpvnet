import glob, os
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


from model.preprocess_sample import preprocess_xy_images


def get_augmented(
        X_train,
        Y_train,
        batch_size=32,
        seed=0,
        data_gen_args=dict(
            rotation_range=10.0,
            height_shift_range=0.02,
            shear_range=5,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode="constant",
        )
    ):
    """
    Copied from keras_unet: https://github.com/karolzak/keras-unet
    Duplicated here just to reduce dependencies

    Note that because get_augmented is only used for the training data, this
    may be one cause of validation loss being lower than training loss.

    Parameters
    ----------
    X_train
    Y_train
    batch_size
    seed
    data_gen_args

    Returns
    -------

    """
    # Train data, provide the same seed and keyword arguments to the fit and
    # flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(
        X_train, batch_size=batch_size, shuffle=True, seed=seed
    )
    Y_train_augmented = Y_datagen.flow(
        Y_train, batch_size=batch_size, shuffle=True, seed=seed
    )

    train_generator = zip(X_train_augmented, Y_train_augmented)
    return train_generator


# Questions:
#   - Should we try tuning the learning rate:
#       https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
#   - Should we use Freeze_encoder? Batchnorm? Monte Carlo Dropout?


def train_unet(input_dir, mask_dir, log_file, weight_file, final_weight_file,
               backbone="resnet34", seed=42, imsize=576, val_frac=0.1):
    """

    Parameters
    ----------
    input_dir
    mask_dir
    log_file
    weight_file
    final_weight_file
    backbone
    seed
    imsize
    val_frac

    Returns
    -------

    """
    # Get the list of all input/output files
    images = glob.glob(os.path.join(input_dir, "*.png"))
    masks = glob.glob(os.path.join(mask_dir, "*.png"))

    # Load and split the data
    print("==== Load and Split Data ====")
    x, y = preprocess_xy_images(images, masks, (imsize, imsize))
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_frac,
                                                      random_state=seed)
    del x, y  # Free up memory

    assert y_val.shape == x_val.shape
    assert y_train.shape == x_train.shape
    print("x_train: ", x_train.shape)
    print("x_val: ", x_val.shape)

    # Preprocess the inputs via segmentation_model
    print("==== Preprocess Data ====")
    preprocess_input = sm.get_preprocessing(backbone)
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    print("==== Augment Data ====")
    # Augment the data
    # Memory problems on this step
    # For more fixes see:
    #   https://github.com/keras-team/keras/issues/1627
    #   https://stackoverflow.com/questions/46705600/keras-fit-image-augmentations-to-training-data-using-flow-from-directory
    train_gen = get_augmented(
        x_train,
        y_train,
        seed=seed,
        batch_size=4,  # 2
        data_gen_args=dict(
            rotation_range=30.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            # shear_range=50,
            zoom_range=0.2,
            # horizontal_flip=True,
            # vertical_flip=True,
            # fill_mode='constant'
        )
    )

    # Setup outputs
    print("==== Setup Callbacks ====")
    model_filename = weight_file
    checkpoint_callback = ModelCheckpoint(
        model_filename,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )
    csv_logger_callback = CSVLogger(log_file, append=True, separator=',')

    # Create the model
    print("==== Create Model ====")
    model = sm.Unet(backbone,
                    encoder_weights='imagenet',
                    input_shape=(imsize, imsize, 3),
                    classes=1,
                    decoder_use_batchnorm=False,
                    encoder_freeze = True)
    print("==== Compile Model ====")
    model.compile(
        optimizer=SGD(lr=0.0009, momentum=0.99),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    print("==== Train ====")
    history = model.fit(
        train_gen,
        steps_per_epoch=100,
        epochs=350,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_callback, csv_logger_callback],
        verbose=2
    )

    if final_weight_file is not None:
        model.save_weights(final_weight_file)


if __name__ == '__main__':
    mysize = 576
    myseed = 42
    mybackbone = "resnet34"
    myinputpath = "c:\\nycdata\\sample_dataset_tiles\\"
    mymaskpath = "c:\\nycdata\\sample_dataset_mask_tiles\\"
    myweightfile = "c:\\nycdata\\test_weights.h5"
    myfinalweightfile = "c:\\nycdata\\test_weights_final.h5"
    mylogfile = "c:\\nycdata\\test_log.csv"

    train_unet(myinputpath, mymaskpath, mylogfile, myweightfile, myfinalweightfile, mybackbone, myseed, mysize)