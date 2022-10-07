import json
import os
import glob

from PIL import Image


mydir = 'C:\\nycdata\\boro_queens_sp18_png\\'


def generate_blank_json_file(json_file, shape):
    with open(json_file, "w") as file:
        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(json_file).replace(".json", ".png"),
            "imageData":None,
            "imageHeight": shape[1],
            "imageWidth": shape[0]
        }
        json_str = json.dumps(data, indent=2)
        file.write(json_str)


def generate_blank_json_dir(somedir):
    files = [f for f in glob.glob(somedir + "*.png")]
    for f in files:
        jsonfn = os.path.join(os.path.dirname(f), os.path.basename(f).replace(".png",".json"))
        if not os.path.exists(jsonfn):
            with Image.open(f) as im:
                shape = im.size
            generate_blank_json_file(jsonfn, shape)



if __name__ == "__main__":
    generate_blank_json_dir(mydir)