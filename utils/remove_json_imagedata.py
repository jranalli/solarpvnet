import json
import os
import glob


mydir = 'C:\\nycdata\\boro_queens_sp18_png\\'


def clear_imageData_file(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    if data['imageData'] is not None:
        with open(json_file, "w") as file:
            data['imageData'] = None
            json_str = json.dumps(data, indent=2)
            file.write(json_str)


def clear_imageData_dir(somedir):
    files = [f for f in glob.glob(somedir + "*.json")]
    for f in files:
        clear_imageData_file(f)



if __name__ == "__main__":
    clear_imageData_dir(mydir)