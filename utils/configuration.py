import importlib


def get_loop_iter(something):
    """
    Wrapper to simplify checking if tqdm is installed

    Usage:
        for file_name in get_loop_iter(file_names):
            # Operate On file_name

    Parameters
    ----------
    something: object
        Anything you want to iterate over in a loop.

    Returns
    -------
    loopiter
        tqdm(something) if tqdm installed, otherwise something
    """
    if importlib.util.find_spec("tqdm"):
        from tqdm import tqdm
        loopiter = tqdm(something)
    else:
        loopiter = something

    return loopiter
