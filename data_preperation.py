import pandas as pd
import torch
from torch.utils.data import DataLoader

from pathmanager import get_dataset_path
from rayTune_common.constants import random_seed


def split_data():
    """
    Method to be run a single time
    Splits the dataset into training, test and validation set and stores the sets as separate csv files
    :return:
    """
    # Read the dataset from csv file
    path = get_dataset_path("well_data.csv")
    df = pd.read_csv(path, index_col=0)

    # Test set (this is the period for which we must estimate QTOT)
    test_set = df.iloc[2000:2500]

    # Make a copy of the dataset and remove the test dataset
    train_val_set = df.copy().drop(test_set.index)

    # Sample validation dataset without replacement (10%)
    val_set = train_val_set.sample(frac=0.1, replace=False, random_state=random_seed)

    # The remaining dataset is used for training (90%)
    train_set = train_val_set.copy().drop(val_set.index)

    # Check that the numbers add up
    n_points = len(train_set) + len(val_set) + len(test_set)
    assert (n_points == len(df))

    # Write training set to csv
    path = get_dataset_path("training_set.csv")
    train_set.to_csv(path)

    # Write validation set to csv
    path = get_dataset_path("validation_set.csv")
    val_set.to_csv(path)

    # Write test set to csv
    path = get_dataset_path("test_set.csv")
    test_set.to_csv(path)


def prepare_data(
        INPUT_COLS: [],
        OUTPUT_COLS: [],
        train_batch_size: int
) -> (DataLoader, torch.Tensor, torch.Tensor, DataLoader, torch.Tensor, torch.Tensor):
    """
    Prepares the dataset to be used for HPO
    Converts to torch tensors and dataset loaders
    :param INPUT_COLS: list of strings
    :param OUTPUT_COLS: list of strings
    :return:
    :return: train_loader, x_val, y_val, val_loader, x_test, y_test
    """
    # INPUT_COLS = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
    # OUTPUT_COLS = ['QTOT']

    path = get_dataset_path("training_set.csv")
    train_set = pd.read_csv(path, index_col=0)
    path = get_dataset_path("validation_set.csv")
    val_set = pd.read_csv(path, index_col=0)
    path = get_dataset_path("test_set.csv")
    test_set = pd.read_csv("/home/knut/Documents/project/dataset/test_set.csv", index_col=0)

    # Get input and output tensors and convert them to torch tensors
    x_train = torch.from_numpy(train_set[INPUT_COLS].values).to(torch.float)
    y_train = torch.from_numpy(train_set[OUTPUT_COLS].values).to(torch.float)

    x_val = torch.from_numpy(val_set[INPUT_COLS].values).to(torch.float)
    y_val = torch.from_numpy(val_set[OUTPUT_COLS].values).to(torch.float)

    # Create dataset loaders
    # Here we specify the batch size and if the dataset should be shuffled
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_set), shuffle=False)

    # Get input and output as torch tensors
    x_test = torch.from_numpy(test_set[INPUT_COLS].values).to(torch.float)
    y_test = torch.from_numpy(test_set[OUTPUT_COLS].values).to(torch.float)

    return train_loader, x_val, y_val, val_loader, x_test, y_test


split_data()
