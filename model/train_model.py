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

    Parameters
    ----------
    X_train
    Y_train
    X_val
    Y_val
    batch_size
    seed
    data_gen_args

    Returns
    -------

    """
    # Train data, provide the same seed and keyword arguments to the fit and flow methods
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

MODEL_IMAGE_SIZE = 576
seed = 42
backbone = "resnet34"
inputpath = "c:\\nycdata\\sample_dataset_tiles\\"
maskpath = "c:\\nycdata\\sample_dataset_mask_tiles\\"
weightfile = "c:\\nycdata\\test_weights.h5"
logfile = "c:\\nycdata\\test_log.csv"

def main():
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument('train_data', help='input annotated directory')
    # parser.add_argument('weight_file', help='weight file')
    # parser.add_argument('train_log', help='training log file')
    # parser.add_argument('--backbone', help='model backbone', default="resnet34")
    # parser.add_argument('--seed', help='random seed for cross validation', default=0)
    # parser.add_argument('--final_weight_file', help='final weight file', default=None)
    # args = parser.parse_args()

    size = MODEL_IMAGE_SIZE

    # Get the list of all input/output files
    orgs = glob.glob(inputpath + "*.png")
    masks = glob.glob(maskpath + "*.png")

    # Load and split the data
    x, y = preprocess_sample.preprocess_xy_images(orgs, masks, (size, size))
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=seed)
    del x, y  # Free up memory

    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)

    # Preprocess the inputs via segmentation_model
    preprocess_input = sm.get_preprocessing(backbone)
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # Augment the data
    # Memory problems on this step
    # For more fixes see:
    #   https://github.com/keras-team/keras/issues/1627
    #   https://stackoverflow.com/questions/46705600/keras-fit-image-augmentations-to-training-data-using-flow-from-directory
    train_gen = get_augmented(
        x_train,
        y_train,
        seed=seed,
        batch_size=4, #2
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
    print("Setup Callback")
    model_filename = weightfile
    checkpoint_callback = ModelCheckpoint(
        model_filename,
        verbose=1,
        monitor='val_loss',
        save_best_only=True,
    )
    csv_logger_callback = CSVLogger(logfile, append=True, separator=',')

    # Create the model
    print("Create Model")
    model = sm.Unet(backbone,
                    encoder_weights='imagenet',
                    input_shape=(size, size, 3),
                    classes=1,
                    decoder_use_batchnorm=False,
                    encoder_freeze = True)
    print("Compile Model")
    model.compile(
        optimizer=SGD(lr=0.0009, momentum=0.99),
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    print("Train")
    history = model.fit(
        train_gen,
        steps_per_epoch=100,
        epochs=350,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint_callback, csv_logger_callback],
        verbose=2
    )


    model.save_weights(model_filename.final_weight_file)

if __name__ == '__main__':
    main()