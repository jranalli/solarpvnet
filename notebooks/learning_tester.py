from model.dataset_manipulation import split_test_train
from model.train_model import train_unet
from model.eval_model import eval_model
import os
import gc

import numpy as np
import random
from tensorflow import random as tfr


def run_A():
    ### Train #####
    my_test_ratio = 0.2
    myseed = 42
    mysize = 576
    mybackbone = "resnet34"
    subset = 0
    subset_seed = myseed
    epochs = 200

    weightss = ['best', 'final']

    freezes = [True]
    batchnorms = [True]
    patiences = [10]

    for weights in weightss:
        for freeze in freezes:
            for patience in patiences:
                for batchnorm in batchnorms:

                    drive = "f:"


                    np.random.seed(myseed)
                    random.seed(myseed)
                    tfr.set_seed(myseed)


                    pathroot = os.path.join(drive, 'solardnn')

                    site = "Germany"

                    runroot = os.path.join(pathroot, site)

                    resultroot = os.path.join(runroot, "results_lrn_demo")

                    dataroot = os.path.join(runroot, "tile_subsets")
                    this_run_root = os.path.join(dataroot, f"set{subset}_seed{subset_seed}")
                    img_root = os.path.join(dataroot, f"img_set{subset}_seed{subset_seed}")
                    mask_root = os.path.join(dataroot, f"mask_set{subset}_seed{subset_seed}")


                    trn_im, trn_msk, tst_im, tst_msk = split_test_train(img_root, mask_root,
                                                                        this_run_root,
                                                                        test_ratio=my_test_ratio,
                                                                        seed=myseed)


                    myweightfile = os.path.join(resultroot, f"demo_norm{batchnorm}_freeze{freeze}_patience{patience}_weights_best.h5")
                    myfinalweightfile = os.path.join(resultroot, f"demo_norm{batchnorm}_freeze{freeze}_patience{patience}_weights_final.h5")
                    mylogfile = os.path.join(resultroot, f"demo_norm{batchnorm}_freeze{freeze}_patience{patience}_trainlog.csv")


                    train_unet(trn_im, trn_msk, mylogfile, myweightfile, myfinalweightfile,
                               mybackbone, myseed, (mysize, mysize),
                               epochs=epochs, freeze_encoder=freeze, patience=patience)

                    if weights == 'best':
                        weight_to_use = myweightfile
                    elif weights == 'final':
                        weight_to_use = myfinalweightfile

                    mypreddir = os.path.join(resultroot, f"demo_norm{batchnorm}_{weights}_freeze{freeze}_patience{patience}\\pred")
                    myplotdir = os.path.join(resultroot, f"demo_norm{batchnorm}_{weights}_freeze{freeze}_patience{patience}\\plots")
                    myresultfile = os.path.join(resultroot, f"demo_norm{batchnorm}_{weights}_freeze{freeze}_patience{patience}\\evallog.csv")

                    eval_model(tst_im, tst_msk, weight_to_use, myresultfile,
                               mypreddir, myplotdir, backbone=mybackbone,
                               img_size=(mysize, mysize))

if __name__ == "__main__":
    run_A()