import os


def get_dataset_path(s: str):
    """
    Gets path to dataset s
    :param s: name of dataset file. e.g: test_set.csv
    :return: absolute path to dataset file
    """
    root = os.path.dirname(os.path.abspath(__file__))
    dpath = os.path.join(root, "dataset")
    spath = os.path.join(dpath, s)
    return spath


def get_results_path():
    root = os.path.dirname(os.path.abspath(__file__))
    rpath = os.path.join(root, "results")
    return rpath
