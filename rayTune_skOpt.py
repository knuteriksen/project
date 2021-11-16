import numpy as np
import skopt
import torch
import torch.utils.data
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.skopt import SkOptSearch

from rayTune_common.constants import random_seed
from rayTune_common.test import test_model
from rayTune_common.train import train
from rayTune_common.utils import get_best_trial, trial_to_model


def optimize(space: [], iterations: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    optimizer = skopt.Optimizer(
        space,
        base_estimator="GP",
        n_initial_points=5,
        acq_func="EI",
        acq_func_kwargs={"xi": 0.05}
    )

    skopt_search = SkOptSearch(
        optimizer=optimizer,
        space=["hidden_layers", "hidden_layer_width", "lr", "l2", "batch_size"],
        metric="mean_square_error",
        mode="min"
    )

    algo = ConcurrencyLimiter(skopt_search, max_concurrent=1)

    result = tune.run(
        tune.with_parameters(train),
        name="skopt",
        metric="mean_square_error",
        mode="min",
        search_alg=algo,
        num_samples=iterations,
        resources_per_trial={"cpu": 1, "gpu": 0},
        verbose=3
    )

    best_trial = get_best_trial(result)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial best mean square error: {}".format(
        best_trial.last_result["mean_square_error"]))

    best_trial_model = trial_to_model(best_trial)
    test_model(model=best_trial_model, batch_size=best_trial.config["batch_size"])
