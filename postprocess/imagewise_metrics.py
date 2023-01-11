import segmentation_models.metrics as m
from segmentation_models.base.functional import round_if_needed
from PIL import Image
import numpy as np
import tensorflow as tf

gt = r"D:\data\solardnn\NY-Q\tiles\mask\002200_62.png"
pt = r"D:\data\solardnn\NY-Q\predictions\NY-Q_resnet34_42_v1_predicting_NY-Q\pred_masks\002200_62.png"
size = 576

t = np.array([0.9])

r = m.Recall(threshold=t)
p = m.Precision(threshold=t)
i = m.IOUScore(threshold=t)
with Image.open(gt) as i1:
    gt = np.array(i1.resize((size, size)))/255
    gta = gt.copy()
    gt = np.expand_dims(gt, 2)
    gt = tf.convert_to_tensor(gt, dtype=np.float32)

with Image.open(pt) as i2:
    pt = np.array(i2)[:,:,0]/255
    pta = pt.copy()
    pt = np.expand_dims(pt, 2)
    pt = tf.convert_to_tensor(pt, dtype=np.float32)



v = r(gt, pt)
p = p(gt, pt)
i = i(gt, pt)

print(f"Recall Keras: {v}")
print(f"Precision Keras: {p}")
print(f"IOU Keras: {i}")


pp = tf.cast(pt>t, tf.float32)
tp = gt*pp

print(f"Recall Man: {np.sum(tp)/np.sum(gt)}")
print(f"Precision Man: {np.sum(tp)/np.sum(pp)}")

gt = gta
pt = pta


tot = np.size(pt)
pp = pt>t

pn = np.size(pt) - np.sum(pp)

tp = np.sum(gt * pp)
fp = np.sum(pp) - tp
fn = np.sum(gt) - tp
tn = pn - fn

print(f"Recall NP: {tp/(tp+fn)}")
print(f"Precision NP: {tp/(tp+fp)}")
print(f"IOU NP: {tp/(tp+fp+fn)}")