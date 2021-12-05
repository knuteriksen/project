import pandas as pd

from rayTune_common.constants import random_seed


def split_data():
    """
    Method to be run a single time
    Splits the dataset into training, test and validation set and stores the sets as separate csv files
    :return:
    """
    # Read the dataset from csv file
    path = "/home/knut/Documents/project/dataset/well_data.csv"
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
    path = "/home/knut/Documents/project/dataset/training_set.csv"
    train_set.to_csv(path)

    # Write validation set to csv
    path = "/home/knut/Documents/project/dataset/validation_set.csv"
    val_set.to_csv(path)

    # Write test set to csv
    path = "/home/knut/Documents/project/dataset/test_set.csv"
    test_set.to_csv(path)
