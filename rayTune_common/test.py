"""
This file contains the different methods for testing the performance of the neural network after HPO
"""
import pandas as pd
import torch

from data_preperation import prepare_data
from rayTune_common.constants import ins, outs


def test_model(model, batch_size):
    """

    :param model: the model to test
    :param batch_size: the batch size used for this configuration
    :return:
    """

    # Import traing, validation and test data
    train_loader, x_val, y_val, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=ins,
        OUTPUT_COLS=outs,
        train_batch_size=int(batch_size)
    )

    model.eval()

    # Make prediction
    pred_test = model(x_test)

    # Calculate MSE
    mse_test = torch.mean(torch.pow(pred_test - y_test, 2))

    return mse_test.item()


def test_training_error(model, batch_size):
    """

    :param model: the model to test
    :param batch_size: the batch size used for this configuration
    :return:
    """

    path = "dataset/training_set.csv"
    train_set = pd.read_csv(path, index_col=0)

    x_train = torch.from_numpy(train_set[ins].values).to(torch.float)
    y_train = torch.from_numpy(train_set[outs].values).to(torch.float)

    model.eval()

    # Make prediction
    pred_test = model(x_train)

    # Calculate MSE
    mse_test = torch.mean(torch.pow(pred_test - y_train, 2))

    return mse_test.item()


def test_validation_error(model, batch_size):
    """

    :param model: the model to test
    :param batch_size: the batch size used for this configuration
    :return:
    """

    # Import traing, validation and test data
    train_loader, x_val, y_val, val_loader, x_test, y_test = prepare_data(
        input_cols=ins,
        output_cols=outs,
        train_batch_size=int(batch_size)
    )

    model.eval()

    # Make prediction
    pred_test = model(x_val)

    # Calculate MSE
    mse_test = torch.mean(torch.pow(pred_test - y_val, 2))

    return mse_test.item()
