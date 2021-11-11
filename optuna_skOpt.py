import optuna
import torch
import numpy as np

from optuna.integration import SkoptSampler
from optuna.trial import Trial

from common.constants import random_seed
from data_preperation import prepare_data


class Net(torch.nn.Module):
    """
    PyTorch offers several ways to construct neural networks.
    Here we choose to implement the network as a Module class.
    This gives us full control over the construction and clarifies our intentions.
    """

    def __init__(self, inputs: int, outputs: int, trial: Trial):
        """

        :param inputs:
        :param outputs:
        :param trial:
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        hidden_layers = trial.suggest_int("hidden_layers", 2, 5, 1)
        hidden_layer_width = trial.suggest_int("hidden_layer_width", 40, 60, 1)

        layers = [inputs] + [hidden_layer_width]*hidden_layers + [outputs]

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

        # Modules/layers must be registered to enable saving of notebooks
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
        Save notebooks state
        :param path: Path to save notebooks state
        :return: None
        """
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

    def load(self, path: str):
        """
        Load notebooks state from file
        :param path: Path to saved notebooks state
        :return: None
        """
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(checkpoint['model_state_dict'])


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

    net = Net(
        len(inputs),
        len(outputs),
        trial
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
