import utils.fileio
from model.dataset_manipulation import test_train_valid_split_list, make_combo_dataset_txt
from model.train_model import train_unet
from model.eval_model import eval_model, pr_curve

from postprocess.images_to_plots import multimodel_plots, model_boundary_plot, single_case_plot
from postprocess.summarize_results import generate_run_summary
from postprocess.imagewise_metrics import aggregate_imagewise_metrics
import os
import gc
from collections import OrderedDict


def run():
    # #### SETTINGS #####

    # ## Global ##
    dataroot = os.path.join("d:", "data", 'solardnn')

    test_sets = ["CA-F", "CA-S", "FR-G", "FR-I", "DE-G", "NY-Q"]
    train_sets = test_sets + ["CMB-6",
                              "CMB-5-CA-F", "CMB-5-CA-S", "CMB-5-FR-G", "CMB-5-FR-I", "CMB-5-DE-G", "CMB-5-NY-Q",
                              "CMB-2-NYQ-CAF", "CMB-2-NYQ-CAS", "CMB-2-NYQ-FRG", "CMB-2-NYQ-FRI", "CMB-2-NYQ-DEG",
                              "CMB-2-CAF-CAS", "CMB-2-CAF-FRG", "CMB-2-CAF-FRI", "CMB-2-CAF-DEG", "CMB-2-CAF-NYQ",
                              "CMB-2-CAS-CAF", "CMB-2-CAS-FRG", "CMB-2-CAS-FRI", "CMB-2-CAS-DEG", "CMB-2-CAS-NYQ",
                              "CMB-2-FRG-CAF", "CMB-2-FRG-CAS", "CMB-2-FRG-FRI", "CMB-2-FRG-DEG", "CMB-2-FRG-NYQ",
                              "CMB-2-FRI-CAF", "CMB-2-FRI-CAS", "CMB-2-FRI-FRG", "CMB-2-FRI-DEG", "CMB-2-FRI-NYQ",
                              "CMB-2-DEG-CAF", "CMB-2-DEG-CAS", "CMB-2-DEG-FRG", "CMB-2-DEG-FRI", "CMB-2-DEG-NYQ",]
    
    combo_sets = {"CMB-6": ["CA-F", "CA-S", "FR-G", "FR-I", "DE-G", "NY-Q"],
                  "CMB-5-CA-F": ["CA-S", "FR-G", "FR-I", "DE-G", "NY-Q"],  # Excludes CA-F
                  "CMB-5-CA-S": ["CA-F", "FR-G", "FR-I", "DE-G", "NY-Q"],  # Excludes CA-S
                  "CMB-5-FR-G": ["CA-F", "CA-S", "FR-I", "DE-G", "NY-Q"],  # Excludes FR-G
                  "CMB-5-FR-I": ["CA-F", "CA-S", "FR-G", "DE-G", "NY-Q"],  # Excludes FR-I
                  "CMB-5-DE-G": ["CA-F", "CA-S", "FR-G", "FR-I", "NY-Q"],  # Excludes DE-G
                  "CMB-5-NY-Q": ["CA-F", "CA-S", "FR-G", "FR-I", "DE-G"],  # Excludes NY-Q

                  "CMB-2-NYQ-CAF": ["NY-Q", "CA-F"],
                  "CMB-2-NYQ-CAS": ["NY-Q", "CA-S"],
                  "CMB-2-NYQ-FRG": ["NY-Q", "FR-G"],
                  "CMB-2-NYQ-FRI": ["NY-Q", "FR-I"],
                  "CMB-2-NYQ-DEG": ["NY-Q", "DE-G"],

                  "CMB-2-CAF-CAS": ["CA-F", "CA-S"],
                  "CMB-2-CAF-FRG": ["CA-F", "FR-G"],
                  "CMB-2-CAF-FRI": ["CA-F", "FR-I"],
                  "CMB-2-CAF-DEG": ["CA-F", "DE-G"],
                  "CMB-2-CAF-NYQ": ["CA-F", "NY-Q"],

                  "CMB-2-CAS-CAF": ["CA-S", "CA-F"],
                  "CMB-2-CAS-FRG": ["CA-S", "FR-G"],
                  "CMB-2-CAS-FRI": ["CA-S", "FR-I"],
                  "CMB-2-CAS-DEG": ["CA-S", "DE-G"],
                  "CMB-2-CAS-NYQ": ["CA-S", "NY-Q"],

                  "CMB-2-FRG-CAF": ["FR-G", "CA-F"],
                  "CMB-2-FRG-CAS": ["FR-G", "CA-S"],
                  "CMB-2-FRG-FRI": ["FR-G", "FR-I"],
                  "CMB-2-FRG-DEG": ["FR-G", "DE-G"],
                  "CMB-2-FRG-NYQ": ["FR-G", "NY-Q"],

                  "CMB-2-FRI-CAF": ["FR-I", "CA-F"],
                  "CMB-2-FRI-CAS": ["FR-I", "CA-S"],
                  "CMB-2-FRI-FRG": ["FR-I", "FR-G"],
                  "CMB-2-FRI-DEG": ["FR-I", "DE-G"],
                  "CMB-2-FRI-NYQ": ["FR-I", "NY-Q"],

                  "CMB-2-DEG-CAF": ["DE-G", "CA-F"],
                  "CMB-2-DEG-CAS": ["DE-G", "CA-S"],
                  "CMB-2-DEG-FRG": ["DE-G", "FR-G"],
                  "CMB-2-DEG-FRI": ["DE-G", "FR-I"],
                  "CMB-2-DEG-NYQ": ["DE-G", "NY-Q"],
                  }

    # ## Dataset ##
    do_build_datasets = True

    splits = [0.2, 0.72, 0.08]  # Test, Train, Valid
    myseeds = [42, 2023]
    n_set = 1000

    # ## Train ##
    do_train_models = True

    mybackbones = ["resnet34"]
    mysize = 576
    epochs = 200
    patience = 10
    norm = True
    freeze = True
    model_revs = ["1", "2"]

    # ## Test ##
    do_test_models = True
    test_weights = 'best'

    # ## Post ##
    do_post = True

    do_summary = True
    do_single_plots = False
    do_boundary_plots = False
    do_multi_plots = True
    do_imagewise_metrics = True

    # #### END SETTINGS ####





    print("\n\n===== BUILD PATHS =====\n\n")
    paths = configure_paths(dataroot, train_sets, myseeds, mybackbones, model_revs, test_sets)

    if do_build_datasets:
        print("\n\n===== BUILD DATASETS =====\n\n")
        build_datasets(paths, train_sets, myseeds, n_set, splits, combo_sets)

    if do_train_models:
        print("\n\n===== TRAINING =====\n\n")
        train_models(paths, train_sets, myseeds, mybackbones, model_revs, mysize, epochs, freeze, patience, norm)

    if do_test_models:
        print("\n\n===== TESTING =====\n\n")
        eval_models(paths, train_sets, myseeds, mybackbones, model_revs, test_sets, mysize, norm, test_weights)

    if do_post:
        print("\n\n===== POST =====\n\n")
        postprocess(paths, train_sets, myseeds, mybackbones, model_revs, test_sets, do_summary, do_single_plots, do_boundary_plots, do_multi_plots, do_imagewise_metrics)


def configure_paths(data_root_dir, train_sets, seeds, backbones, model_revs, test_sets):
    """
    Generate a paths data object holding the conventional paths for the project.

    Nested dicts of paths[train_set][seed][backbone][model_rev][test_set] with things at various levels
        [train_set]
            Stores directory paths germane to the different training datasets
            - tiles: the root tile directory for the dataset.  e.g. d:/solardnn/NY-Q/tiles
            - img_root: the root image directory for dataset. e.g. d:/solardnn/NY-Q/tiles/img
            - mask_root: the root mask directory for dataset. e.g. d:/solardnn/NY-Q/tiles/mask
            - negative_list: the text file containing the list of tiles that are blank. e.g. d:/solardnn/NY-Q/tiles/negative_tiles.txt
            - positive_list: the text file containing the list of tiles with objects. e.g. d:/solardnn/NY-Q/tiles/positive_tiles.txt
            - model_out_root: directory where models should be saved. e.g. d:/solardnn/NY-Q/models
            - prediction_root: directory for predictions when the model is tested. e.g. d:/solardnn/NY-Q/predictions
        [seed]
            Stores the various dataset definition files. These will be located in the tiles directory for the training set.
            - test_im: Location of dataset definition file for test images. e.g. d:/solardnn/NY-Q/tiles/test_im_42.txt
            - test_mask: Location of dataset definition file for test masks. e.g. d:/solardnn/NY-Q/tiles/test_mask_42.txt
            - train_im: Location of dataset definition file for train images. e.g. d:/solardnn/NY-Q/tiles/train_im_42.txt
            - train_mask: Location of dataset definition file for train masks. e.g. d:/solardnn/NY-Q/tiles/train_mask_42.txt
            - valid_im: Location of dataset definition file for valid images. e.g. d:/solardnn/NY-Q/tiles/valid_im_42.txt
            - valid_mask: Location of dataset definition file for valid masks. e.g. d:/solardnn/NY-Q/tiles/valid_mask_42.txt
        backbone:
            Holder for next level
        revision:
            Stores the info about the trained model. All model outputs are stored together in the models directory for
            the training set. The model filenames differentiate them.
            - best_weights: Location of best weights file. e.g. d:/solardnn/NY-Q/models/NY-Q_resnet34_42_v1_weights_best.h5
            - final_weights: Location of final weights file. e.g. d:/solardnn/NY-Q/models/NY-Q_resnet34_42_v1_weights_final.h5
            - train_log: Location of log file from training. e.g. d:/solardnn/NY-Q/models/NY-Q_resnet34_42_v1_trainlog.csv
            - result_root: Location of global results when using any model with these configurations. e.g. d:/solardnn/results/resnet34_42_v1
            - summary_file: Location of global results Excel summary when any model with these configurations. e.g. d:/solardnn/results/resnet34_42_v1/resnet34_42_v1_summary.xlsx
            ** Note: result_root and summary_file don't depend on the train set at all, because they are computed across multiple train sets
        test_set:
            Stores info about the outputs when we test a model. Stored nested below the predictions root dir for the
            training set. The subdir name will always be /{train_set}_{backbone}_{seed}_v{model_rev}_predicting_{test_set}/
            - prediction_dir: The location of the prediction masks dir for the test set.
                e.g. d:/solardnn/NY-Q/predictions/NY-Q_resnet34_42_v1_predicting_CA-F/pred_masks
            - plot dir: The location of plot files if they're generated.
                e.g. d:/solardnn/NY-Q/predictions/NY-Q_resnet34_42_v1_predicting_CA-F/plots
            - result file: The location of the datafile for the prediction summary
                e.g. d:/solardnn/NY-Q/predictions/NY-Q_resnet34_42_v1_predicting_CA-F/NY-Q_resnet34_42_v1_predicting_CA-F_data.csv
            - boundary_plot_root: Location of global boundary plot results when using this test set. e.g. d:/solardnn/results/resnet34_42_v1/CA-F_test/boundary_plot
            - multi_plot_root: Location of global multi-model plot results when using this test set. e.g. d:/solardnn/results/resnet34_42_v1/CA-F_test/multi_plot
            ** Note: boundary_plot_root and multi_plot_root don't depend on the train set at all, because they are computed across multiple train sets
            ** Thus, they can be called for the test_set in both set slots. i.e. paths[test_set][seed][backbone][model_rev][test_set]['boundary_plot_root']


    Parameters
    ----------
    data_root_dir: str
        Root directory for all the data. Sites should be subdirs here and must contain subdirectories of tiles, images and masks.
        e.g. d:/data/solardnn/{SITE}/tiles/img   and   d:/data/solardnn/{SITE}/tiles/mask
    train_sets: list[str]
        List of strings representing all the training sets to consider. e.g. ['CA-F','NY-Q','CMB-6']
    seeds: list[int]
        List of all seeds to consider. e.g. [42]
    backbones: list[str]
        List of all backbones to run. e.g. ['resnet34','resnet50']
    model_revs: list[str]
        List of revision flags to add to the models. Note that these will be appended to filenames but will have no impact on the runs.
        Technically could probably be any type.
        e.g. [1]
    test_sets: list[str]
        List of strings representing all the test data sets to consider. These should never include combos.
        e.g. ['CA-F','NY-Q']

    Returns
    -------
    Deeply nested dictionary following conventions notated above.
        paths[train_set][seed][backbone][model_rev][test_set]
    """
    paths = {}

    for train_set in train_sets:
        paths[train_set] = {}
        siteroot = os.path.join(data_root_dir, train_set)
        tileroot = os.path.join(data_root_dir, train_set, "tiles")
        negativelist = os.path.join(tileroot, "negative_tiles.txt")
        positivelist = os.path.join(tileroot, "positive_tiles.txt")
        modeloutroot = os.path.join(siteroot, "models")
        predictionroot = os.path.join(siteroot, "predictions")

        if "CMB" in train_set:  # let the img_root be pulled from the file
            img_root = None
            mask_root = None
        else:
            img_root = os.path.join(tileroot, f"img")
            mask_root = os.path.join(tileroot, f"mask")

        paths[train_set]['tiles'] = tileroot
        paths[train_set]['img_root'] = img_root
        paths[train_set]['mask_root'] = mask_root
        paths[train_set]['negative_list'] = negativelist
        paths[train_set]['positive_list'] = positivelist
        paths[train_set]['model_out_root'] = modeloutroot
        paths[train_set]['prediction_root'] = predictionroot

        for seed in seeds:
            paths[train_set][seed] = {}

            paths[train_set][seed]['test_im'] = os.path.join(tileroot, f"test_img_{seed}.txt")
            paths[train_set][seed]['test_mask'] = os.path.join(tileroot, f"test_mask_{seed}.txt")
            paths[train_set][seed]['train_im'] = os.path.join(tileroot, f"train_img_{seed}.txt")
            paths[train_set][seed]['train_mask'] = os.path.join(tileroot, f"train_mask_{seed}.txt")
            paths[train_set][seed]['valid_im'] = os.path.join(tileroot, f"valid_img_{seed}.txt")
            paths[train_set][seed]['valid_mask'] = os.path.join(tileroot, f"valid_mask_{seed}.txt")

            for backbone in backbones:
                paths[train_set][seed][backbone] = {}
                for model_rev in model_revs:
                    paths[train_set][seed][backbone][model_rev] = {}
                    paths[train_set][seed][backbone][model_rev]['best_weights'] = os.path.join(modeloutroot,
                                                                                               f"{train_set}_{backbone}_{seed}_v{model_rev}_weights_best.h5")
                    paths[train_set][seed][backbone][model_rev]['final_weights'] = os.path.join(modeloutroot,
                                                                                                f"{train_set}_{backbone}_{seed}_v{model_rev}_weights_final.h5")
                    paths[train_set][seed][backbone][model_rev]['train_log'] = os.path.join(modeloutroot,
                                                                                            f"{train_set}_{backbone}_{seed}_v{model_rev}_trainlog.csv")

                    global_result_root_dir = os.path.join(data_root_dir, fr"results\{backbone}_{seed}_{model_rev}")
                    summary_file = os.path.join(global_result_root_dir, rf"{backbone}_{seed}_v{model_rev}_summary.xlsx")
                    paths[train_set][seed][backbone][model_rev]['result_root'] = global_result_root_dir
                    paths[train_set][seed][backbone][model_rev]['summary_file'] = summary_file

                    for test_set in test_sets:
                        paths[train_set][seed][backbone][model_rev][test_set] = {}
                        test_result_subdir = os.path.join(predictionroot, f"{train_set}_{backbone}_{seed}_v{model_rev}_predicting_{test_set}")
                        paths[train_set][seed][backbone][model_rev][test_set]['prediction_dir'] = os.path.join(test_result_subdir, "pred_masks")
                        paths[train_set][seed][backbone][model_rev][test_set]['plot_dir'] = os.path.join(test_result_subdir, "plots")
                        paths[train_set][seed][backbone][model_rev][test_set]['result_file'] = os.path.join(test_result_subdir, f"{train_set}_{backbone}_{seed}_v{model_rev}_predicting_{test_set}_data.csv")
                        paths[train_set][seed][backbone][model_rev][test_set]['prcurve_file'] = os.path.join(test_result_subdir, f"{train_set}_{backbone}_{seed}_v{model_rev}_predicting_{test_set}_prcurve.csv")

                        boundary_plot_dir = os.path.join(global_result_root_dir, rf"{test_set}_test", "boundary_plot")
                        multi_plot_dir = os.path.join(global_result_root_dir, rf"{test_set}_test", "multi_plot")

                        paths[train_set][seed][backbone][model_rev][test_set]['boundary_plot_root'] = boundary_plot_dir
                        paths[train_set][seed][backbone][model_rev][test_set]['multi_plot_root'] = multi_plot_dir
                        paths[train_set][seed][backbone][model_rev][test_set]['imagewise_metric_file'] = os.path.join(global_result_root_dir, rf"{test_set}_test", f"{backbone}_{seed}_v{model_rev}_{test_set}_imgmetrics.xlsx")

    return paths


def build_datasets(paths, train_sets, seeds, n_set, test_train_valid, combo_sets=None):
    """
    Wrapper for building the test/train/validation subsets that are listed in text files.

    Some special note is warranted for combo datasets. They must use the word CMB in the train_set identifier. If CMB
    datasets are present, the optional parameter `combo_sets` must be provided. It should be a dictionary, keyed by the
    train_set identifier for the combo set, and then valued as a list of other train_sets to use. All values specified
    within the `combo_set` must be listed as a train_set in `paths`.
    e.g.
        train_sets = ["NY-Q", "DE-G", "FR-I", "CMB2"]
        combo_sets = {"CMB2": ["NY-Q", "FR-I"]}

    Parameters
    ----------
    paths: nested dict
        Output from configure_paths(). See docs for configure_paths() for a description.
    train_sets: list[str]
        List of strings representing all the training sets to consider. e.g. ['CA-F','NY-Q','CMB-6']
    seeds: list[int]
        List of all seeds to consider. e.g. [42]
    n_set: int
        The # of images to include in the total dataset. Can be set to None to generate list of all files (not compatible
        combo datasets).
    test_train_valid: list[float, float, float]
        floats representing the test/train/validation split. e.g. [0.1, 0.8, 0.1]
    combo_sets: dict
        dictionary defining any CMB datasets. See full description for info.
    """
    for train_set in train_sets:
        for seed in seeds:
            if "CMB" in train_set:
                try:
                    these_sets = combo_sets[train_set]

                    # File names
                    tr_im_f = paths[train_set][seed]['train_im']
                    tr_m_f = paths[train_set][seed]['train_mask']
                    v_im_f = paths[train_set][seed]['valid_im']
                    v_m_f = paths[train_set][seed]['valid_mask']

                    # Paths for the base sets
                    all_img_rt = [paths[someset]['img_root'] for someset in these_sets]
                    all_mask_rt = [paths[someset]['mask_root'] for someset in these_sets]

                    # train img
                    all_tr_i_fs = [paths[someset][seed]['train_im'] for someset in these_sets]
                    make_combo_dataset_txt(all_tr_i_fs, tr_im_f, all_img_rt, total_imgs=int(n_set * test_train_valid[1]),
                                           seed=seed)

                    # train mask
                    all_tr_m_fs = [paths[someset][seed]['train_mask'] for someset in these_sets]
                    make_combo_dataset_txt(all_tr_m_fs, tr_m_f, all_mask_rt, total_imgs=int(n_set * test_train_valid[1]),
                                           seed=seed)

                    # val img
                    all_v_i_fs = [paths[someset][seed]['valid_im'] for someset in these_sets]
                    make_combo_dataset_txt(all_v_i_fs, v_im_f, all_img_rt, total_imgs=int(n_set * test_train_valid[2]), seed=seed)

                    # val mask
                    all_v_m_fs = [paths[someset][seed]['valid_mask'] for someset in these_sets]
                    make_combo_dataset_txt(all_v_m_fs, v_m_f, all_mask_rt, total_imgs=int(n_set * test_train_valid[2]), seed=seed)
                except KeyError:
                    print(f"combo_sets does not contain a specifier for {train_set}. Skipping...")
            else:
                # Build datasets with positive examples only
                tiledir = paths[train_set]['tiles']
                positives = utils.fileio.read_file_list(paths[train_set]['positive_list'])
                test_train_valid_split_list(positives, positives, tiledir, test_train_valid=test_train_valid, seed=seed, n_set=n_set)


def train_models(paths, train_sets, seeds, backbones, model_revs, img_size, epochs, freeze_encoder, patience, batchnorm):
    """
    Wrapper to help calling the training for multiple models at once

    Parameters
    ----------
    paths: nested dict
        Output from configure_paths(). See docs for configure_paths() for a description.
    train_sets: list[str]
        List of strings representing all the training sets to consider. e.g. ['CA-F','NY-Q','CMB-6']
    seeds: list[int]
        List of all seeds to consider. e.g. [42]
    backbones: list[str]
        List of all backbones to run. e.g. ['resnet34','resnet50']
    model_revs: list[str]
        List of revision flags to add to the models. Note that these will be appended to filenames but will have no impact on the runs.
        Technically could probably be any type.
        e.g. [1]
    img_size: int
        Size images should be resized to. Images will be resized to (img_size x img_size)
    epochs: int
        Max number of epochs
    freeze_encoder: bool
        Should the encoder be frozen?
    patience: int
        Patience that should be used in early stopping. Set to 0 to run for full epochs.
    batchnorm: bool
        Should batch normalization be used?
    """
    for train_set in train_sets:
        for seed in seeds:
            for backbone in backbones:
                for model_rev in model_revs:
                    print("============")
                    print(f"Training: \nSet: {train_set}\nSeed: {seed}\nBackbone: {backbone}\nRev: v{model_rev}\n")

                    imdir = paths[train_set]['img_root']
                    maskdir = paths[train_set]['mask_root']
                    tr_im_f = paths[train_set][seed]['train_im']
                    tr_m_f = paths[train_set][seed]['train_mask']
                    v_im_f = paths[train_set][seed]['valid_im']
                    v_m_f = paths[train_set][seed]['valid_mask']
                    best_wgt = paths[train_set][seed][backbone][model_rev]['best_weights']
                    final_wgt = paths[train_set][seed][backbone][model_rev]['final_weights']
                    log = paths[train_set][seed][backbone][model_rev]['train_log']

                    train_unet(imdir, maskdir, tr_im_f, tr_m_f, v_im_f, v_m_f, log_file=log, best_weight_file=best_wgt,
                               end_weight_file=final_wgt, backbone=backbone, seed=seed, img_size=(img_size, img_size),
                               epochs=epochs, freeze_encoder=freeze_encoder, patience=patience, batchnorm=batchnorm)

                    gc.collect()


def eval_models(paths, train_sets, seeds, backbones, model_revs, test_sets, img_size, batchnorm, weight_type):
    """
    Wrapper to help perform the model evaluation for a large set of models

    Parameters
    ----------
    paths: nested dict
        Output from configure_paths(). See docs for configure_paths() for a description.
    train_sets: list[str]
        List of strings representing all the training sets to consider. e.g. ['CA-F','NY-Q','CMB-6']
    seeds: list[int]
        List of all seeds to consider. e.g. [42]
    backbones: list[str]
        List of all backbones to run. e.g. ['resnet34','resnet50']
    model_revs: list[str]
        List of revision flags to add to the models. Note that these will be appended to filenames but will have no impact on the runs.
        Technically could probably be any type.
        e.g. [1]
    test_sets: list[str]
        List of strings representing all the test data sets to consider. These should never include combos.
        e.g. ['CA-F','NY-Q']
    img_size: int
        Size images should be resized to. Images will be resized to (img_size x img_size)
    batchnorm: bool
        Should batch normalization be used?
    weight_type: str
        One of 'best' or 'final'. Which weights file should be read for the trained model.
    """
    for train_set in train_sets:
        for seed in seeds:
            for backbone in backbones:
                for model_rev in model_revs:
                    for test_set in test_sets:
                        print("============")
                        print(f"Testing: \nModel: {train_set}\nSeed: {seed}\nBackbone: {backbone}\nRev: v{model_rev}\nTest Set: {test_set}\n")

                        tst_im_f = paths[test_set][seed]['test_im']
                        tst_m_f = paths[test_set][seed]['test_mask']

                        imdir = paths[test_set]['img_root']
                        maskdir = paths[test_set]['mask_root']

                        if weight_type == 'best':
                            wgt_file = paths[train_set][seed][backbone][model_rev]['best_weights']
                        elif weight_type == 'final':
                            wgt_file = paths[train_set][seed][backbone][model_rev]['final_weights']
                        else:
                            print("Weights not found!")
                            continue

                        pred_dir = paths[train_set][seed][backbone][model_rev][test_set]['prediction_dir']
                        res_file = paths[train_set][seed][backbone][model_rev][test_set]['result_file']
                        prcurve_file = paths[train_set][seed][backbone][model_rev][test_set]['prcurve_file']

                        eval_model(imdir, maskdir, tst_im_f, tst_m_f, wgt_file, res_file, pred_dir,
                                   backbone=backbone, img_size=(img_size, img_size), batchnorm=batchnorm)

                        pr_curve(maskdir, tst_m_f, pred_dir, prcurve_file)

                        gc.collect()


def postprocess(paths, train_sets, seeds, backbones, model_revs, test_sets, gen_summary=True, gen_single_plots=False, gen_boundary_plots=False, gen_multi_plots=False, gen_imagewise_metrics=False):
    """
        Wrapper to help perform the postprocessing for a large set of models

        Parameters
        ----------
        paths: nested dict
            Output from configure_paths(). See docs for configure_paths() for a description.
        train_sets: list[str]
            List of strings representing all the training sets to consider. e.g. ['CA-F','NY-Q','CMB-6']
        seeds: list[int]
            List of all seeds to consider. e.g. [42]
        backbones: list[str]
            List of all backbones to run. e.g. ['resnet34','resnet50']
        model_revs: list[str]
            List of revision flags to add to the models. Note that these will be appended to filenames but will have no impact on the runs.
            Technically could probably be any type.
            e.g. [1]
        test_sets: list[str]
            List of strings representing all the test data sets to consider. These should never include combos.
            e.g. ['CA-F','NY-Q']
        gen_summary: bool (default True)
            Should global summary file be generated?
        gen_single_plots: bool (default False)
            Should single case boundary plots be generated?
        gen_boundary_plots: bool (default False)
            Should global boundary comparison plots be generated?
        gen_multi_plots: bool (default False)
            Should global multi_plot representations be generated?
        gen_imagewise_metrics: bool (default False)
            Should imagewise metrics be calculated?
        """
    for seed in seeds:
        for backbone in backbones:
            for model_rev in model_revs:
                if gen_summary:
                    # These happen at a global level assuming multiple models all trained the same way, but spanning
                    # all train sets against all test sets

                    # Build the nested list of files that will be summarized
                    res_files = OrderedDict()
                    summary_file = paths[train_sets[0]][seed][backbone][model_rev]['summary_file']
                    for test_set in test_sets:
                        res_files[test_set] = OrderedDict()
                        for train_set in train_sets:
                            res_files[test_set][train_set] = paths[train_set][seed][backbone][model_rev][test_set][
                                'result_file']
                    # Do the summary
                    generate_run_summary(res_files, summary_file)

                # The summary plots occur for an entire test set but encompass multiple training sets
                for test_set in test_sets:
                    test_img_path = paths[test_set]['img_root']
                    test_mask_path = paths[test_set]['mask_root']

                    # Build a list of the prediction directories for all the training sets (this is an input)
                    pred_dirs = []
                    for train_set in train_sets:
                        pred_dir = paths[train_set][seed][backbone][model_rev][test_set]['prediction_dir']
                        pred_dirs.append(pred_dir)

                        if gen_single_plots:  # These occur at both the training and test set level
                            tst_im_f = paths[test_set][seed]['test_im']
                            tst_m_f = paths[test_set][seed]['test_mask']
                            imdir = paths[test_set]['img_root']
                            maskdir = paths[test_set]['mask_root']
                            plot_dir = paths[train_set][seed][backbone][model_rev][test_set]['plot_dir']
                            single_case_plot(imdir, maskdir, tst_im_f, tst_m_f, pred_dir, plot_dir)

                    # Generate the plots
                    if gen_boundary_plots:
                        # All train_sets have the same directories paths, so just use test_set as the train_set
                        bnddir = paths[test_set][seed][backbone][model_rev][test_set]['boundary_plot_root']

                        print(f"\n=={test_set} BORDER==")
                        model_boundary_plot(test_img_path, test_mask_path, pred_dirs, bnddir, model_names=train_sets)

                    if gen_multi_plots:
                        # All train_sets have the same directories paths, so just use test_set as the train_set
                        multidir = paths[test_set][seed][backbone][model_rev][test_set]['multi_plot_root']

                        print(f"\n=={test_set} COMBO==")
                        multimodel_plots(test_img_path, test_mask_path, pred_dirs, multidir, model_names=train_sets)

                    # Run the metrics
                    if gen_imagewise_metrics:
                        print(f"\n=={test_set} Imagewise==")
                        imgmetricfile = paths[test_set][seed][backbone][model_rev][test_set]['imagewise_metric_file']
                        aggregate_imagewise_metrics(test_mask_path, pred_dirs, imgmetricfile, model_names=train_sets)


if __name__ == "__main__":
    run()
