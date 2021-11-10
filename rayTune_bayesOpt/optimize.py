import os

import numpy as np

import torch
import torch.utils.data

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch

from common.data_preperation import prepare_data
from common.constants import random_seed


class Net(torch.nn.Module):
    """
    PyTorch offers several ways to construct neural networks.
    Here we choose to implement the network as a Module class.
    This gives us full control over the construction and clarifies our intentions.
    """

    def __init__(self, inputs: int, hidden_layers: float, hidden_layer_width: float, outputs: int):
        """

        :param inputs:
        :param hidden_layers:
        :param hidden_layer_width:
        :param outputs:
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        _hl = int(hidden_layers)
        _hlw = int(hidden_layer_width)

        layers = [inputs] + [_hlw]*_hl + [outputs]

        assert len(layers) >= 2, "At least two layers are required (incl. input and output layer)"
        self.layers = layers

        # Fully connected linear layers
        linear_layers = []

        for i in range(len(self.layers) - 1):
            n_in = self.layers[i]
            n_out = self.layers[i + 1]
            layer = torch.nn.Linear(n_in, n_out)

            # Initialize weights and biases
            a = 1 if i == 0 else 2
            layer.weight.data = torch.randn((n_out, n_in)) * np.sqrt(a / n_in)
            layer.bias.data = torch.zeros(n_out)

            # Add to list
            linear_layers.append(layer)

        # Modules/layers must be registered to enable saving of model
        self.linear_layers = torch.nn.ModuleList(linear_layers)

        # Non-linearity (e.g. ReLU, ELU, or SELU)
        self.act = torch.nn.ReLU(inplace=False)

    def forward(self, input):
        """
        Forward pass to evaluate network for input values
        :param input: tensor assumed to be of size (batch_size, n_inputs)
        :return: output tensor
        """
        x = input
        for l in self.linear_layers[:-1]:
            x = l(x)
            x = self.act(x)

        output_layer = self.linear_layers[-1]
        return output_layer(x)

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str):
        """
        Save model state
        :param path: Path to save model state
        :return: None
        """
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

    def load(self, path: str):
        """
        Load model state from file
        :param path: Path to saved model state
        :return: None
        """
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(checkpoint['model_state_dict'])


def train(config, checkpoint_dir=None):
    """

    :param config:
    :param checkpoint_dir:
    :return:
    """
    inputs = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
    outputs = ['QTOT']
    net = Net(
        len(inputs),
        config["hidden_layers"],
        config["hidden_layer_width"],
        len(outputs)
    )

    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net)
    net.to(device)
    """

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Import traing, validation and test data
    train_loader, x_valid, y_valid, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=inputs,
        OUTPUT_COLS=outputs,
        train_batch_size=int(config["batch_size"])
    )

    # Train Network
    _n_epochs = int(config["n_epochs"])
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

            cost = batch_mse + config["l2"]/1000.0 * reg_loss

            # Backward propagation to compute gradient
            cost.backward()

            # Update parameters using gradient
            optimizer.step()

        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)
        print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

        tune.report(mean_square_error=mse_val)

    print("Finished Training")


def test_best_model(best_trial):
    inputs = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
    outputs = ['QTOT']

    best_trained_model = Net(
        len(inputs),
        best_trial.config["hidden_layers"],
        best_trial.config["hidden_layer_width"],
        len(outputs)
    )

    device = "cpu"

    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    # Import traing, validation and test data
    train_loader, x_val, y_val, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=inputs,
        OUTPUT_COLS=outputs,
        train_batch_size=int(best_trial.config["batch_size"])
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
    torch.manual_seed(random_seed)

    config = {
        "n_epochs": tune.uniform(70, 130),
        "l2": tune.uniform(2, 256),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.uniform(8, 16),
        "hidden_layers": tune.uniform(2, 10),
        "hidden_layer_width": tune.uniform(40, 60)
    }

    bayesopt = BayesOptSearch(
        random_search_steps=4,
        utility_kwargs={
            "kind": "ei",
            "xi": 0.001,
            "kappa": 2.5
        }
    )

    result = tune.run(
        tune.with_parameters(train),
        name="Test Bayes Opt",
        metric="mean_square_error",
        mode="min",
        search_alg=bayesopt,
        num_samples=50,
        config=config,
        resources_per_trial={"cpu": 1, "gpu": 0}
    )

    best_trial = result.get_best_trial("mean_square_error", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["mean_square_error"]))

    test_best_model(best_trial=best_trial)


if __name__ == "__main__":
    main()
