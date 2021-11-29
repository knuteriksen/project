import os

import torch
from ray.tune import ExperimentAnalysis
from ray.tune.trial import Trial

from model.model import Net
from rayTune_common.constants import ins, outs


def get_best_trial(result: ExperimentAnalysis):
    """

    :param result: Result after experiment
    :return: The result as a Trial object
    """
    return result.get_best_trial("mean_square_error", "min", "all")


def trial_to_model(trial: Trial):
    """
    Converts a Trial object into a model of Net
    :param trial: the Trial object to convert
    :return: A trained version of Net
    """
    model = Net(
        len(ins),
        int(trial.config["hidden_layers"]),
        int(trial.config["hidden_layer_width"]),
        len(outs),
        dropout_value=trial.config["dropout"]
    )

    device = "cpu"
    # model = model.eval()
    model.to(device)

    checkpoint_path = os.path.join(trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    return model


def config_to_model(config: {}, checkpoint_path: str):
    model = Net(
        len(ins),
        int(config["hidden_layers"]),
        int(config["hidden_layer_width"]),
        len(outs),
        dropout_value=config["dropout"]
    )

    device = "cpu"
    # model = model.eval()
    model.to(device)
    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    return model
