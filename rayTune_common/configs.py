"""
This file contains the different configuration spaces for the HPO search
"""

from ray import tune

config6 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 1),
    "hidden_layer_width": tune.quniform(30, 120, 10),
    "dropout": tune.choice([0, 1]),
    "batch_size": tune.choice([4, 8, 16, 32, 64]),
    "l2": tune.uniform(1e-3, 1e-1),
}

config5 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 1),
    "hidden_layer_width": tune.quniform(30, 120, 10),
    "dropout": tune.choice([0, 1]),
    "batch_size": tune.choice([4, 8, 16, 32, 64]),
    "l2": 1e-2,
}

config4 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 1),
    "hidden_layer_width": tune.quniform(30, 120, 10),
    "dropout": tune.choice([0, 1]),
    "batch_size": 16,
    "l2": 1e-2,
}

config3 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 1),
    "hidden_layer_width": tune.quniform(30, 120, 10),
    "dropout": 1,
    "batch_size": 16,
    "l2": 1e-2,
}

config2 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 1),
    "hidden_layer_width": 70,
    "dropout": 1,
    "batch_size": 16,
    "l2": 1e-2,
}

config1 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": 5,
    "hidden_layer_width": 70,
    "dropout": 1,
    "batch_size": 16,
    "l2": 1e-2,
}
