import json

from fileio import files_of_type


def clear_imagedata(json_file):
    """
    Remove imagedata from labelme JSON files. By default labelme stores a full
    copy of the image in the JSON file leading to huge JSON files.

    Parameters
    ----------
    json_file: str
        Full path to json file that will be modified in place
    """
    with open(json_file, "r") as file:
        data = json.load(file)

    if data['imageData'] is not None:
        print("Removing data from: " + json_file)
        with open(json_file, "w") as file:
            data['imageData'] = None
            json_str = json.dumps(data, indent=2)
            file.write(json_str)


# Example directory for testing
mydir = 'C:\\nycdata\\boro_queens_sp18_png\\'

if __name__ == "__main__":
    for fn in files_of_type(mydir, "*.json"):
        clear_imagedata(fn)
