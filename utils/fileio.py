import os
import glob


def verify_dir(mydir):
    """
    Check if directory exists. If not, create its full path.

    Parameters
    ----------
    mydir: str
        full path of directory to check for
    """
    if not os.path.exists(mydir):
        os.mkdir(mydir)


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
    return [f for f in glob.glob(mydir + searchstr)]
