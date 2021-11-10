import numpy as np

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

import torch
import torch.utils.data

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from common.data_preperation import prepare_data

"""
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from rayTune_bayesOpt.data_preperation import prepare_data
from rayTune_bayesOpt.constants import random_seed
"""

best_mse = 100

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


space = [
        Integer(2, 5, name="hidden_layers"),
        Integer(2, 5, name="hidden_layer_width"),
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

    print("*************************")
    print("Training with params:")
    print(params)
    print("*************************")

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

        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)

        if mse_val < best_mse:
            best_mse = mse_val
            torch.save(
                (net.state_dict(), optimizer.state_dict()),
                "/home/knut/Documents/TTK28-Courseware-master/results/best"
            )
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Current best params")
            print(params)
            print("New MSE: ", mse_val)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

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
    model_state, optimizer_state = torch.load("/home/knut/Documents/TTK28-Courseware-master/results/best")
    best_trained_model.load_state_dict(model_state)

    # Import traing, validation and test data
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

    res_gp = gp_minimize(objective, space, n_calls=10, random_state=1, acq_func="EI", verbose=True)

    print(res_gp.x)

    test_best_model(res_gp.x)


if __name__ == "__main__":
    main()
