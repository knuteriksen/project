import os

import torch
from ray.tune import Analysis

from data_preperation import prepare_data
from model.model import Net
from rayTune_common.constants import ins, outs

path = "/home/knut/ray_results/b_000"
list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
list_subfolders_with_paths.sort(key=lambda x: x.split("_")[4])

for i, sfolder in enumerate(list_subfolders_with_paths):
    list_checkpoint_with_path = [f.path for f in os.scandir(sfolder) if f.is_dir()]
    list_checkpoint_with_path.sort(key=lambda x: x.split("_")[-1])
    cpath = os.path.join(list_checkpoint_with_path[-1], "checkpoint")

    anal = Analysis(sfolder, default_metric="mean_square_error", default_mode="min")
    config = anal.get_best_config(metric="mean_square_error", mode="min")

    best_trained_model = Net(
        len(ins),
        int(config["hidden_layers"]),
        int(config["hidden_layer_width"]),
        len(outs),
        dropout_value=config["dropout"]
    )

    device = "cpu"

    best_trained_model.to(device)

    model_state, optimizer_state = torch.load(cpath)
    best_trained_model.load_state_dict(model_state)

    # Import traing, validation and test data
    train_loader, x_val, y_val, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=ins,
        OUTPUT_COLS=outs,
        train_batch_size=int(config["batch_size"])
    )

    best_trained_model.eval()

    # Predict on validation data
    pred_val = best_trained_model(x_val)

    # Compute MSE, MAE and MAPE on validation data
    print('Error on validation data')

    mse_val = torch.mean(torch.pow(pred_val - y_val, 2))
    print(f'MSE: {mse_val.item()}')

    """
    mae_val = torch.mean(torch.abs(pred_val - y_val))
    print(f'MAE: {mae_val.item()}')

    mape_val = 100 * torch.mean(torch.abs(torch.div(pred_val - y_val, y_val)))
    print(f'MAPE: {mape_val.item()} %')
    """
    # Make prediction
    pred_test = best_trained_model(x_test)

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
