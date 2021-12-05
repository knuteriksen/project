from ray import tune

config5 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 2),
    "hidden_layer_width": tune.quniform(30, 120, 15),
    "dropout": tune.uniform(0.0, 0.5),
    "l2": tune.loguniform(1e-6, 1e-1),
}

config4 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 2),
    "hidden_layer_width": tune.quniform(30, 120, 15),
    "dropout": tune.uniform(0.0, 0.5),
    "l2": 1e-3,
}

config3 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 2),
    "hidden_layer_width": tune.quniform(30, 120, 15),
    "dropout": 0.2,
    "l2": 1e-3,
}

config2 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": tune.quniform(2, 10, 2),
    "hidden_layer_width": 75,
    "dropout": 0.2,
    "l2": 1e-3,
}

config1 = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "hidden_layers": 6,
    "hidden_layer_width": 70,
    "dropout": 0.2,
    "l2": 1e-3,
}
