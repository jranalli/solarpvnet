import os
import json
import glob

import numpy as np

import PIL.Image

from labelme import utils

from convert_to_png import verify_dir


target_dir = 'C:\\nycdata\\boro_queens_sp18_png\\'
ot_dir = 'C:\\nycdata\\boro_queens_sp18_out\\'


def json_to_binary_file(json_file, out):

    if out is None:
        out_dir = os.path.basename(json_file).replace(".", "_")
        out_dir = os.path.join(os.path.dirname(json_file), out_dir)
    else:
        out_dir = out

    verify_dir(out_dir)

    data = json.load(open(json_file))
    imshape = (data['imageHeight'], data['imageWidth'])

    label_name_to_value = {"_background_": 0, "maybe": 0, "notpv": 0, "pv": 255}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(imshape, data["shapes"], label_name_to_value)
    lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
    lbl_pil.save(os.path.join(out_dir, os.path.basename(json_file).replace(".json", ".png")))

def json_to_binary_dir(png_dir, out):
    json_files = [f for f in glob.glob(png_dir + "*.json")]
    for f in json_files:
        json_to_binary_file(f, out)


if __name__ == "__main__":
    json_to_binary_dir(target_dir, ot_dir)
