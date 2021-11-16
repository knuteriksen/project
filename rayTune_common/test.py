"""
This file contains the different methods for testing the performance of the neural network after HPO
"""

import torch

from data_preperation import prepare_data
from model.model import Net
from rayTune_common.constants import ins, outs


def test_model(model: Net, batch_size):
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

    """
    # Predict on validation data
    pred_val = model(x_val)

    # Compute MSE, MAE and MAPE on validation data
    print('Error on validation data')

    mse_val = torch.mean(torch.pow(pred_val - y_val, 2))
    print(f'MSE: {mse_val.item()}')

    mae_val = torch.mean(torch.abs(pred_val - y_val))
    print(f'MAE: {mae_val.item()}')

    mape_val = 100 * torch.mean(torch.abs(torch.div(pred_val - y_val, y_val)))
    print(f'MAPE: {mape_val.item()} %')
    """
    # Make prediction
    pred_test = model(x_test)

    # Compute MSE, MAE and MAPE on test data
    print('Error on test data')

    mse_test = torch.mean(torch.pow(pred_test - y_test, 2))
    print(f'MSE: {mse_test.item()}')

    """
    mae_test = torch.mean(torch.abs(pred_test - y_test))
    print(f'MAE: {mae_test.item()}')

    mape_test = 100 * torch.mean(torch.abs(torch.div(pred_test - y_test, y_test)))
    print(f'MAPE: {mape_test.item()} %')
    """

    return mse_test.item()
