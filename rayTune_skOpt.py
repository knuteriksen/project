import skopt
import torch
import torch.utils.data
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.skopt import SkOptSearch

from rayTune_common.constants import random_seed
from rayTune_common.test_loop import test_best_model
from rayTune_common.training_loop import train


def optimize(space: [], iterations: int):
    torch.manual_seed(random_seed)

    optimizer = skopt.Optimizer(
        space,
        base_estimator="GP",
        n_initial_points=5,
        acq_func="EI"
    )

    skopt_search = SkOptSearch(
        optimizer=optimizer,
        space=["hidden_layers", "hidden_layer_width", "lr", "l2", "batch_size"],
        metric="mean_square_error",
        mode="min",

    )

    algo = ConcurrencyLimiter(skopt_search, max_concurrent=1)

    result = tune.run(
        tune.with_parameters(train),
        name="Test SkOpt",
        metric="mean_square_error",
        mode="min",
        search_alg=algo,
        num_samples=iterations,
        resources_per_trial={"cpu": 1, "gpu": 0},
        verbose=3
    )

    best_trial = result.get_best_trial("mean_square_error", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["mean_square_error"]))

    test_best_model(best_trial=best_trial)
