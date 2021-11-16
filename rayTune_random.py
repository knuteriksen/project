import numpy as np
import torch
import torch.utils.data
from ray import tune

from rayTune_common.constants import random_seed
from rayTune_common.test import test_model
from rayTune_common.train import train
from rayTune_common.utils import get_best_trial, trial_to_model


def optimize(config: {}, iterations: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    result = tune.run(
        tune.with_parameters(train),
        config=config,
        name="random",
        metric="mean_square_error",
        mode="min",
        num_samples=iterations,
        resources_per_trial={"cpu": 8, "gpu": 0},
        max_concurrent_trials=8,
        verbose=3
    )

    best_trial = get_best_trial(result)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial best mean square error: {}".format(
        best_trial.last_result["mean_square_error"]))

    best_trial_model = trial_to_model(best_trial)
    test_model(model=best_trial_model, batch_size=best_trial.config["batch_size"])
