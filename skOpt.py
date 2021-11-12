import os

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

import torch
import torch.utils.data

from data_preperation import prepare_data
from pathmanager import get_results_path

from model.model import Net


best_mse = 100

space = [
        Integer(2, 5, name="hidden_layers"),
        Integer(40, 60, name="hidden_layer_width"),
        Real(10**-5, 10**0, "log-uniform", name='lr'),
        Real(10**-3, 10**0, "log-uniform", name='l2'),
        Categorical([8, 10, 12], name="batch_size")
    ]


@use_named_args(space)
def objective(**params):
    """
    :param config:
    :return:
    """
    global best_mse
    inputs = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
    outputs = ['QTOT']

    print("*" * 105)
    print(("*" * 41), "Training with params:", ("*" * 41))
    print("*", params, "*")
    print("*" * 105)

    net = Net(
        inputs=len(inputs),
        hidden_layers=params.get("hidden_layers"),
        hidden_layer_width=params.get("hidden_layer_width"),
        outputs=len(outputs)
    )

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=params.get("lr"))

    # Import traing, validation and test data
    train_loader, x_valid, y_valid, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=inputs,
        OUTPUT_COLS=outputs,
        train_batch_size=int(params.get("batch_size"))
    )

    mse_val = 0

    # Train Network
    _n_epochs = 100
    for epoch in range(_n_epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients (from last iteration)
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)

            # Compute cost function
            batch_mse = criterion(outputs, labels)

            reg_loss = 0
            for param in net.parameters():
                reg_loss += param.pow(2).sum()

            cost = batch_mse + params.get("l2") * reg_loss

            # Backward propagation to compute gradient
            cost.backward()

            # Update parameters using gradient
            optimizer.step()

        # Evaluate notebooks on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)

        if mse_val < best_mse:
            best_mse = mse_val
            torch.save(
                (net.state_dict(), optimizer.state_dict()),
                os.path.join(get_results_path(), "best2")
            )
            print("~" * 105)
            print(("~" * 42), "Current best params", ("~" * 42))
            print("~", params, "~")
            print(("~" * 44), "New MSE: {:.4f}".format(mse_val), ("~" * 42))
            print("~" * 105)

    print("Finished Training")
    return mse_val


def test_best_model(result):
    inputs = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
    outputs = ['QTOT']

    best_trained_model = Net(
        len(inputs),
        result[0],
        result[1],
        len(outputs)
    )

    device = "cpu"
    best_trained_model.to(device)
    model_state, optimizer_state = torch.load(os.path.join(get_results_path(), "best2"))
    best_trained_model.load_state_dict(model_state)

    # Import training, validation and test data
    train_loader, x_val, y_val, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=inputs,
        OUTPUT_COLS=outputs,
        train_batch_size=int(result[4])
    )

    # Predict on validation data
    pred_val = best_trained_model(x_val)

    # Compute MSE, MAE and MAPE on validation data
    print('Error on validation data')

    mse_val = torch.mean(torch.pow(pred_val - y_val, 2))
    print(f'MSE: {mse_val.item()}')

    mae_val = torch.mean(torch.abs(pred_val - y_val))
    print(f'MAE: {mae_val.item()}')

    mape_val = 100 * torch.mean(torch.abs(torch.div(pred_val - y_val, y_val)))
    print(f'MAPE: {mape_val.item()} %')

    # Make prediction
    pred_test = best_trained_model(x_test)

    # Compute MSE, MAE and MAPE on test data
    print('Error on test data')

    mse_test = torch.mean(torch.pow(pred_test - y_test, 2))
    print(f'MSE: {mse_test.item()}')

    mae_test = torch.mean(torch.abs(pred_test - y_test))
    print(f'MAE: {mae_test.item()}')

    mape_test = 100 * torch.mean(torch.abs(torch.div(pred_test - y_test, y_test)))
    print(f'MAPE: {mape_test.item()} %')


def main():
    # Random seed
    torch.manual_seed(12345)

    global best_mse
    best_mse = 100

    res_gp = gp_minimize(objective, space, n_calls=10, n_initial_points=5, random_state=1, acq_func="EI", verbose=True)

    print(res_gp.x)

    test_best_model(res_gp.x)

    plot_convergence(res_gp)


if __name__ == "__main__":
    main()
