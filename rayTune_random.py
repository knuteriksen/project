import numpy as np
import torch
import torch.utils.data
from ray import tune

from rayTune_common.constants import random_seed
from rayTune_common.test_loop import test_best_model
from rayTune_common.training_loop import train


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

    best_trial = result.get_best_trial("mean_square_error", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["mean_square_error"]))

    test_best_model(best_trial=best_trial)
