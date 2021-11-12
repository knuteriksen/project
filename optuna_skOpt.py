import optuna
import torch
from optuna.integration import SkoptSampler
from optuna.trial import Trial

from data_preperation import prepare_data
from model.model import Net
from rayTune_common.constants import random_seed


def objective(trial: Trial):
    """

    :param trial:
    :return:
    """
    inputs = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
    outputs = ['QTOT']

    lr = trial.suggest_loguniform("lr", 1e-5, 1)
    l2 = trial.suggest_uniform("l2", 1e-3, 1)
    batch_size = trial.suggest_categorical("batch_size", [8, 10, 12])
    hidden_layers = trial.suggest_int("hidden_layers", 2, 5, 1)
    hidden_layer_width = trial.suggest_int("hidden_layer_width", 40, 60, 1)

    net = Net(
        len(inputs),
        hidden_layers,
        hidden_layer_width,
        len(outputs)
    )

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr)

    # Import traing, validation and test data
    train_loader, x_valid, y_valid, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=inputs,
        OUTPUT_COLS=outputs,
        train_batch_size=int(batch_size)
    )

    # Train Network
    mse_val = 0

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

            cost = batch_mse + l2 * reg_loss

            # Backward propagation to compute gradient
            cost.backward()

            # Update parameters using gradient
            optimizer.step()

        # Evaluate notebooks on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)

        trial.report(mse_val, epoch)

    return mse_val


def optimize():
    # Random seed
    torch.manual_seed(random_seed)

    optimizer = SkoptSampler(
        skopt_kwargs={
            "base_estimator": "GP",
            "n_initial_points": 5,
            "acq_func": "EI"
        }
    )

    study = optuna.create_study(sampler=optimizer, direction="minimize")
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    optimize()
