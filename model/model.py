"""
This file contains the neural network model
"""

import numpy as np
import torch


class Net(torch.nn.Module):
    def __init__(
            self,
            inputs: int,
            hidden_layers: int,
            hidden_layer_width: int,
            outputs: int,
            dropout_value: float
    ):
        """

        :param dropout_value: Dropout value to use. 0.0 If no dropout is desired
        :param inputs: Number of inputs
        :param hidden_layers: Number of hidden layers
        :param hidden_layer_width: Size of hidden layer
        :param outputs: Number of outputs
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        layers = [inputs] + [hidden_layer_width] * hidden_layers + [outputs]

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

            # Add possible dropout
            if dropout_value:
                linear_layers.append(torch.nn.Dropout(dropout_value))

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
