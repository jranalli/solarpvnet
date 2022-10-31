import os
import glob
import shutil


def verify_dir(target_dir):
    """
    Check if directory exists. If not, create its full path.

    Parameters
    ----------
    target_dir: str
        full path of directory to check for
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def clear_dir(target_dir, confirm=True):
    """
    Clear tree of a directory and recreate

    Parameters
    ----------
    target_dir: str
        full path of the directory to clear

    confirm: bool (default True)
        Should a confirmation prompt be used
    """
    if not os.path.exists(target_dir):
        raise FileNotFoundError("Directory does not exist.")
    else:
        if confirm:  # Give a chance to jump out
            prompt = "Really delete all files in directory? (y/n)"
            if input(prompt).lower() != "y":
                return
        shutil.rmtree(target_dir)
        os.mkdir(target_dir)


def is_dir_empty(test_dir):
    """
    Check if directory is empty. If not,

    Parameters
    ----------
    test_dir: str
        full path of directory to check

    Returns
    -------
    bool: True if the directory is empty, False if not
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError("Directory does not exist.")
    elif len(os.listdir(test_dir)) > 0:  # it's not empty
        return False
    else:
        return True


def files_of_type(search_dir: object, search_str: object) -> object:
    """
    Get all files in a directory that match a search string

    Parameters
    ----------
    search_dir: str
        full path of directory to scan
    search_str: str
        The search string to use. Examples: "*.json" or "*.png"

    Returns
    -------
        List of all files in the dir matching the search string
    """
    return [f for f in glob.glob(os.path.join(search_dir, search_str))]
