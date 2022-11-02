from model.dataset_manipulation import split_test_train
from model.train_model import train_unet
from model.eval_model import eval_model
import os
import gc

### Train #####
my_test_ratio = 0.2
myseeds = [42]
mysize = 576
mybackbones = ["resnet34"]
subset = 0
subset_seed = 42
epochs = 200

drive = "f:"

pathroot = os.path.join(drive, 'solardnn')

sites = [ "Germany", "Cal_Fresno", "Cal_Stockton", "France_ign", "France_google"]  # "Cal_Oxnard" "NYC"- too few files
for site in sites:
    runroot = os.path.join(pathroot, site)
    resultroot = os.path.join(runroot, "results")

    dataroot = os.path.join(runroot, "tile_subsets")
    this_run_root = os.path.join(dataroot, f"set{subset}_seed{subset_seed}")
    img_root = os.path.join(dataroot, f"img_set{subset}_seed{subset_seed}")
    mask_root = os.path.join(dataroot, f"mask_set{subset}_seed{subset_seed}")

    for myseed in myseeds:
        for mybackbone in mybackbones:

            myinputpath = os.path.join(this_run_root, f"train_img_{myseed}")
            mymaskpath = os.path.join(this_run_root, f"train_mask_{myseed}")
            myweightfile = os.path.join(resultroot, f"set{subset}_setseed{subset_seed}_{mybackbone}_{myseed}_weights_best.h5")
            myfinalweightfile = os.path.join(resultroot, f"set{subset}_setseed{subset_seed}_{mybackbone}_{myseed}_weights_final.h5")
            mylogfile = os.path.join(resultroot, f"set{subset}_setseed{subset_seed}_{mybackbone}_{myseed}_trainlog.csv")

            if os.path.exists(myweightfile):
                print(f"Skipping {site}-{mybackbone}-{myseed}")
                continue

            split_test_train(img_root, mask_root, this_run_root,
                             test_ratio=my_test_ratio, seed=myseed)
            train_unet(myinputpath, mymaskpath, mylogfile, myweightfile, myfinalweightfile, mybackbone, myseed, (mysize, mysize), epochs=epochs)

### Eval #####

myseeds = [42]
mysize = 576
mybackbones = ["resnet34"]
subset = 0
subset_seed = 42

drive = "f:"
pathroot = os.path.join(drive, 'solardnn')

sites = ["Germany", "Cal_Fresno", "Cal_Stockton", "France_ign", "France_google"]  # "Cal_Oxnard" "NYC"- too few files
for site in sites:
    for model in sites:
        dataroot = os.path.join(os.path.join(pathroot, site), "tile_subsets")
        this_run_root = os.path.join(dataroot, f"set{subset}_seed{subset_seed}")
        resultroot = os.path.join(pathroot, "results")
        modelroot = os.path.join(os.path.join(pathroot, model), "results")

        for myseed in myseeds:
            for mybackbone in mybackbones:

                myweightfile = os.path.join(modelroot, f"set{subset}_setseed{subset_seed}_{mybackbone}_{myseed}_weights_best.h5")

                myimages = os.path.join(this_run_root, f"test_img_{myseed}")
                mymasks = os.path.join(this_run_root, f"test_mask_{myseed}")

                mypreddir = os.path.join(resultroot, f"results_set{subset}_{site}_predby{model}_{mybackbone}_{myseed}\\pred")
                myplotdir = os.path.join(resultroot, f"results_set{subset}_{site}_predby{model}_{mybackbone}_{myseed}\\plots")
                myresultfile = os.path.join(resultroot, f"results_set{subset}_{site}_predby{model}_{mybackbone}_{myseed}\\results_set{subset}_{site}_predby{model}_{mybackbone}_{myseed}.csv")

                eval_model(myimages, mymasks, myweightfile, myresultfile,
                           mypreddir, myplotdir, backbone=mybackbone,
                           img_size=(mysize, mysize))

                gc.collect()