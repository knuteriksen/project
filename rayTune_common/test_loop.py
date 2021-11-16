"""
This file contains the different methods for testing the performance of the neural network after HPO
"""

import os

import torch
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.trial import Trial

from data_preperation import prepare_data
from model.model import Net
from rayTune_common.constants import ins, outs


def get_best_trial(result: ExperimentAnalysis):
    """

    :param result: Result after experiment
    :return: The result as a Trial object
    """
    return result.get_best_trial("mean_square_error", "min", "all")


def trail_to_model(trial: Trial):
    """
    Converts a Trial object into a model of Net
    :param trial: the Trial object to convert
    :return: A trained version of Net
    """
    model = Net(
        len(ins),
        int(trial.config["hidden_layers"]),
        int(trial.config["hidden_layer_width"]),
        len(outs),
        dropout_value=trial.config["dropout"]
    )

    device = "cpu"
    model.to(device)

    checkpoint_path = os.path.join(trial.checkpoint.value, "checkpoint")
    print(checkpoint_path)

    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    return model


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

    # Make prediction
    pred_test = model(x_test)

    # Compute MSE, MAE and MAPE on test data
    print('Error on test data')

    mse_test = torch.mean(torch.pow(pred_test - y_test, 2))
    print(f'MSE: {mse_test.item()}')

    mae_test = torch.mean(torch.abs(pred_test - y_test))
    print(f'MAE: {mae_test.item()}')

    mape_test = 100 * torch.mean(torch.abs(torch.div(pred_test - y_test, y_test)))
    print(f'MAPE: {mape_test.item()} %')
