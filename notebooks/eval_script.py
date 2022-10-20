from model.preprocess_sample import split_test_train
from model.train_model import train_unet
from model.eval_model import eval_model
import os
import gc

### Train #####
my_test_ratio = 0.1
myseeds = [0]
mysize = 576
mybackbones = ["resnet18"]

pathroot = 'f:\\solardnn'

sites = ["NYC", "Germany"]
for site in sites:
    runroot = os.path.join(pathroot, site)
    resultroot = os.path.join(runroot, "results")

    dataroot = os.path.join(runroot, "tiles")
    img_root = os.path.join(dataroot, "imgs")
    mask_root = os.path.join(dataroot, "masks")

    for myseed in myseeds:
        for mybackbone in mybackbones:

            myinputpath = os.path.join(dataroot, f"train_imgs_{myseed}")
            mymaskpath = os.path.join(dataroot, f"train_masks_{myseed}")
            myweightfile = os.path.join(resultroot, f"{mybackbone}_{myseed}_weights_best.h5")
            myfinalweightfile = os.path.join(resultroot, f"{mybackbone}_{myseed}_weights_final.h5")
            mylogfile = os.path.join(resultroot, f"{mybackbone}_{myseed}_trainlog.csv")

            if os.path.exists(myweightfile):
                print(f"Skipping {site}-{mybackbone}-{myseed}")
                continue

            split_test_train(img_root, mask_root, dataroot, seed=myseed, test_ratio=my_test_ratio)
            train_unet(myinputpath, mymaskpath, mylogfile, myweightfile, myfinalweightfile, mybackbone, myseed, mysize, epochs=200)

### Eval #####

myseeds = [0]
mysize = 576
mybackbones = ["resnet18"]

pathroot = 'f:\\solardnn'

sites = ["NYC", "Germany"]
for site in sites:
    for model in sites:
        dataroot = os.path.join(os.path.join(pathroot, site), "tiles")
        resultroot = os.path.join(pathroot, "results")
        modelroot = os.path.join(os.path.join(pathroot, model), "results")

        for myseed in myseeds:
            for mybackbone in mybackbones:

                myweightfile = os.path.join(modelroot, f"{mybackbone}_{myseed}_weights_best.h5")

                myimages = os.path.join(dataroot, f"test_imgs_{myseed}")
                mymasks = os.path.join(dataroot, f"test_masks_{myseed}")

                mypreddir = os.path.join(resultroot, f"results_{site}_predby{model}_{mybackbone}_{myseed}\\pred")
                myplotdir = os.path.join(resultroot, f"results_{site}_predby{model}_{mybackbone}_{myseed}\\plots")
                myresultfile = os.path.join(resultroot, f"results_{site}_predby{model}_{mybackbone}_{myseed}\\results_{site}_predby{model}_{mybackbone}_{myseed}.csv")

                eval_model(myimages, mymasks, myweightfile, myresultfile,
                           mypreddir, myplotdir, backbone=mybackbone,
                           imsize=mysize)

                gc.collect()