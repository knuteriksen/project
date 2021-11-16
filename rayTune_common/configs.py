"""
This file contains the different configuration spaces for the HPO search
"""

from ray import tune

config6 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([4, 8, 16, 32, 64]),
    "hidden_layers": tune.quniform(2, 10, 1),
    "hidden_layer_width": tune.quniform(30, 60, 5),
    "dropout": tune.choice([0, 1]),
    "l2": tune.uniform(1e-3, 1e-1),
}
