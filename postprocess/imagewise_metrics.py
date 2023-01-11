from PIL import Image
import numpy as np


def compute_imagewise_metrics(truth_file, prediction_file, threshold=0.5):
    """
    Compute metrics for an individual image mask pair

    Parameters
    ----------
    truth_file: str
        full context of ground truth mask
    prediction_file: str
        full context of prediction mask
    threshold: float
        A threshold value that will be applied to determine the level of prediction that constitutes "true".
        Should be between 0-1 as masks will be normed.

    Returns
    -------
    metrics: float
        return iou, precision, recall, f1
    """
    # Load images
    with Image.open(prediction_file) as i2:
        pt = np.array(i2)[:, :, 0]  # Predictions are RGB for some reason
        pt = pt/np.max(pt)  # normalize
    with Image.open(truth_file) as i1:
        gt = np.array(i1.resize(pt.shape))  # Reshape to match prediction
        gt = gt/np.max(gt)

    # Compute Truth Table
    tot = np.size(pt)  # total pix
    pp = pt > threshold  # predicted positives, array
    tp = gt * pp  # true positives, array

    n_pp = np.sum(pp)  # n of predicted positives
    n_pn = tot - n_pp  # n of predicted negatives (all pix - positives)

    n_tp = np.sum(tp)  # n of true positives
    n_fp = n_pp - n_tp  # n of false positives
    n_fn = np.sum(gt) - n_tp  # n of false negatives (actual positives - true positives)
    n_tn = n_pn - n_fn  # n of true negatives

    # Metric Definitions
    recall = n_tp / (n_tp + n_fn)
    precision = n_tp / (n_tp + n_fp)
    iou = n_tp / (n_tp + n_fp + n_fn)
    f1 = 2 * n_tp / (2 * n_tp + n_fp + n_fn)

    return iou, precision, recall, f1

    # # This can also be done with tensorflow, but it took roughly 10x the time of the manual method
    # import tensorflow as tf
    # recall = tf.keras.metrics.Recall(thresholds=[threshold])
    # precis = tf.keras.metrics.Precision(thresholds=[threshold])
    # iou = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])  # 2 classes (0 and 1), target is id 1 of those.
    #
    # # gt = <Normalized Truth Image Array>
    # # pt = <Normalized Prediction Image Array>
    #
    # recall.reset_state() # Not strictly necessary but needed to reset when adding more
    # recall.update_state(gt, pt)
    # print(recall.result().numpy())
    #
    # precis.reset_state()
    # precis.update_state(gt, pt)
    # print(precis.result().numpy())
    #
    # iou.reset_state()
    # iou.update_state(gt, pt > threshold)
    # print(iou.result().numpy())


def run():
    truth_file = r"D:\data\solardnn\NY-Q\tiles\mask\002200_62.png"
    prediction_file = r"D:\data\solardnn\NY-Q\predictions\NY-Q_resnet34_42_v1_predicting_NY-Q\pred_masks\002200_62.png"
    threshold = 0.5

    iou, precision, recall, f1 = compute_imagewise_metrics(truth_file, prediction_file, threshold)
    print(iou)
    print(precision)
    print(recall)
    print(f1)


if __name__ == "__main__":
    run()
