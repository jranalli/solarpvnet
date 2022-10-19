import os
import glob
import shutil


def verify_dir(mydir):
    """
    Check if directory exists. If not, create its full path.

    Parameters
    ----------
    mydir: str
        full path of directory to check for
    """
    if not os.path.exists(mydir):
        os.makedirs(mydir)


def clear_dir(mydir, confirm=True):
    """
    Clear tree of a directory and recreate

    Parameters
    ----------
    mydir: str
        full path of the directory to clear

    confirm: bool (default True)
        Should a confirmation prompt be used
    """
    if not os.path.exists(mydir):
        raise FileNotFoundError("Directory does not exist.")
    else:
        if confirm:  # Give a chance to jump out
            prompt = "Really delete all files in directory? (y/n)"
            if input(prompt).lower() != "y":
                return
        shutil.rmtree(mydir)
        os.mkdir(mydir)


def is_dir_empty(mydir):
    """
    Check if directory is empty. If not,

    Parameters
    ----------
    mydir: str
        full path of directory to check

    Returns
    -------
    bool: True if the directory is empty, False if not
    """
    if not os.path.exists(mydir):
        raise FileNotFoundError("Directory does not exist.")
    elif len(os.listdir(mydir)) > 0:  # it's not empty
        return False
    else:
        return True


def files_of_type(mydir, searchstr):
    """
    Get all files in a directory that match a search string

    Parameters
    ----------
    mydir: str
        full path of directory to scan
    searchstr: str
        The search string to use. Examples: "*.json" or "*.png"

    Returns
    -------
        List of all files in the dir matching the search string
    """
    return [f for f in glob.glob(os.path.join(mydir, searchstr))]
