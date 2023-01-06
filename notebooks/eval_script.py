from model.dataset_manipulation import test_train_valid_split
from model.train_model import train_unet
from model.eval_model import eval_model
import os
import gc

### Train #####
my_test_ratio = 0.2
my_train_ratio = 0.72
n_set = 1000
myseeds = [42]
mysize = 576
mybackbones = ["resnet34"]
subset = 0
subset_seed = 42
epochs = 200

patience = 10
norm = True
freeze = True

drive = "f:"

pathroot = os.path.join(drive, 'solardnn2')

sites = ["Germany", "Cal_Fresno", "Cal_Stockton", "France_ign", "France_google", "NYC", "combo_dataset"]  # "Cal_Oxnard" "NYC"- too few files
sites = ["NYC"]
for site in sites:
    runroot = os.path.join(pathroot, site)
    resultroot = os.path.join(runroot, "results")

    dataroot = os.path.join(runroot, "tiles")
    img_root = os.path.join(dataroot, f"img")
    mask_root = os.path.join(dataroot, f"mask")

    for myseed in myseeds:
        for mybackbone in mybackbones:

            myweightfile = os.path.join(resultroot, f"{mybackbone}_{myseed}_weights_best.h5")
            myfinalweightfile = os.path.join(resultroot, f"{mybackbone}_{myseed}_weights_final.h5")
            mylogfile = os.path.join(resultroot, f"{mybackbone}_{myseed}_trainlog.csv")

            if os.path.exists(myweightfile):
                print(f"Skipping {site}-{mybackbone}-{myseed}")
                continue

            test_im, test_mask, train_im, train_mask, valid_im, valid_mask = test_train_valid_split(img_root, mask_root, dataroot,
                                test_train_valid=[my_test_ratio, my_train_ratio, 1-my_test_ratio-my_train_ratio], seed=myseed, n_set=n_set)
            train_unet(img_root, mask_root, train_im, train_mask, valid_im, valid_mask, log_file=mylogfile, weight_file=myweightfile,
                       end_weight_file=myfinalweightfile, backbone=mybackbone, seed=myseed, img_size=(mysize, mysize),
                       epochs=epochs, freeze_encoder=freeze, batchnorm=norm,
                       patience=patience)

### Eval #####

myseeds = [42]
mysize = 576
mybackbones = ["resnet34"]
subset = 0
subset_seed = 42

weights = 'best'

drive = "f:"
pathroot = os.path.join(drive, 'solardnn2')

sites = ["Germany", "Cal_Fresno", "Cal_Stockton", "France_ign", "France_google", "NYC"]  # "Cal_Oxnard" "NYC"- too few files
# models = sites + ["combo_dataset"]
sites = ["NYC"]
models = sites
for site in sites:
    dataroot = os.path.join(os.path.join(pathroot, site), "tiles")
    img_root = os.path.join(dataroot, f"img")
    mask_root = os.path.join(dataroot, f"mask")

    resultroot = os.path.join(pathroot, "results")

    for model in models:

        modelroot = os.path.join(os.path.join(pathroot, model), "results")

        for myseed in myseeds:
            for mybackbone in mybackbones:

                myimage_file = os.path.join(dataroot, f"test_img_{myseed}.txt")
                mymask_file = os.path.join(dataroot, f"test_mask_{myseed}.txt")

                if weights == 'best':
                    myweightfile = os.path.join(modelroot, f"{mybackbone}_{myseed}_weights_best.h5")
                elif weights == 'final':
                    myweightfile = os.path.join(modelroot, f"{mybackbone}_{myseed}_weights_final.h5")

                mypreddir = os.path.join(resultroot, f"results_{site}_predby{model}_{mybackbone}_{myseed}\\pred")
                # myplotdir = os.path.join(resultroot, f"results_{site}_predby{model}_{mybackbone}_{myseed}\\plots")
                myplotdir = None
                myresultfile = os.path.join(resultroot, f"results_{site}_predby{model}_{mybackbone}_{myseed}\\results_{site}_predby{model}_{mybackbone}_{myseed}.csv")

                eval_model(img_root, mask_root, myimage_file, mymask_file, myweightfile, myresultfile,
                           mypreddir, myplotdir, backbone=mybackbone,
                           img_size=(mysize, mysize), batchnorm=norm)

                gc.collect()