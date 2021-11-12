import csv
import os
from datetime import datetime


def init_new_results_dir(num_hp: int, search_method: str):
    """
    Adds new directory to results folder for this HPO.
    Adds new subdirectory for models for this HPO
    :param search_method: BO or RS
    :param num_hp: number of hyperparameters in configuration space
    :return: path to this hpo result directory, path to models subdirectory,
            path to gpr subdirectory, path to acq subdirectory
    """
    num_hp_ = str(num_hp)

    now = datetime.now()
    now_ = now.strftime("%m_%d_%H_%M_%S")

    parent = '../results/'
    child = now_ + "_" + num_hp_ + "_" + search_method.strip().lower()
    path = os.path.join(parent, child)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    model_parent = path
    model_child = "models"
    model_path = os.path.join(model_parent, model_child)
    try:
        os.mkdir(model_path)
    except OSError:
        print("Creation of the directory %s failed" % model_path)
    else:
        print("Successfully created the directory %s " % model_path)

    gpr_parent = path
    gpr_child = "gpr"
    gpr_path = os.path.join(gpr_parent, gpr_child)

    try:
        os.mkdir(gpr_path)
    except OSError:
        print("Creation of the directory %s failed" % gpr_path)
    else:
        print("Successfully created the directory %s " % gpr_path)

    acq_parent = path
    acq_child = "acq"
    acq_path = os.path.join(acq_parent, acq_child)

    try:
        os.mkdir(acq_path)
    except OSError:
        print("Creation of the directory %s failed" % acq_path)
    else:
        print("Successfully created the directory %s " % acq_path)

    return path, model_path, gpr_path, acq_path


def init_new_iteration_dir(iteration: int, parent: str):
    """
    Adds new directory for this HPO iteration, and final_epoc dir for this iteration
    :param iteration: Which HPO iteration
    :param parent: Path to parent folder, i.e. "../results/mm_dd_HH_MM_SS_xx/models/"
    :return: path to directory for this HPO iteration and path to final_epoc directory for this iteration
    """
    child = str(iteration).rjust(3, "0")
    path = os.path.join(parent, child)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    final_epoc_parent = path
    final_epoc_child = "final_epoc"
    final_epoc_path = os.path.join(final_epoc_parent, final_epoc_child)

    try:
        os.mkdir(final_epoc_path)
    except OSError:
        print("Creation of the directory %s failed" % final_epoc_path)
    else:
        print("Successfully created the directory %s " % final_epoc_path)

    return path, final_epoc_path


def init_new_epoc_dir(e: int, parent: str):
    """
    Adds new directory for this epoc
    :param e: Which epoc
    :param parent: Path to parent folder, i.e. "../results/mm_dd_HH_MM_SS_xx/models/iteration"
    :return: path to directory for this epoc
    """
    child = str(e).rjust(5, "0")
    path = os.path.join(parent, child)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def _write_hpo_selection(hp: [str], parent: str):
    """
    Private function
    Writes to the hpo selection csv file
    :param hp: List of hyperparemter-names, or values
    :param parent: Path to parent folder, i.e  "../results/mm_dd_HH_MM_SS_xx/"
    :return:
    """
    fpath = os.path.join(parent, "hpo_selection.csv")
    with open(fpath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(hp)


def init_hpo_selection_file(hp: [str], parent: str):
    """
    Initializes new hyperparameter selection file with headers
    Intended to be used when initializing new hyperparameter search
    :param hp: List of hyperparemter-names
    :param parent: Path to parent folder, i.e  "../results/mm_dd_HH_MM_SS_xx/"
    :return:
    """
    _write_hpo_selection(hp=hp, parent=parent)


def append_hpo_selection_file(hp: [str], parent: str):
    """
    Appends hyperparameter selections to file
    Intended to be used before each new iteration of HPO
    :param hp: List of hyperparemter-values
    :param parent: Path to parent folder, i.e  "../results/mm_dd_HH_MM_SS_xx/"
    :return:
    """
    _write_hpo_selection(hp=hp, parent=parent)


def append_mse_to_easy_plot(mse: float, parent: str):
    """
    Appends the MSE to the easy_plot file.
    Intended to be used after final epoc of each iteration of HPO
    :param mse: Mean Square Error
    :param parent: Path to parent folder, i.e  "../results/mm_dd_HH_MM_SS_xx/"
    :return:
    """
    fpath = os.path.join(parent, "easy_plot.csv")
    row = [mse]
    with open(fpath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)
