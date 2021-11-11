import torch
import torch.utils.data

from ray import tune
from ray.tune.suggest.skopt import SkOptSearch
import skopt

from common.constants import random_seed

from rayTune_common.training_loop import train
from rayTune_common.test_loop import test_best_model


def optimize():
    # Random seed
    torch.manual_seed(random_seed)

    space = [
        skopt.space.Integer(2, 5, name="hidden_layers"),
        skopt.space.Integer(40, 60, name="hidden_layer_width"),
        skopt.space.Real(10 ** -5, 10 ** 0, "log-uniform", name='lr'),
        skopt.space.Real(10 ** -3, 10 ** 0, "log-uniform", name='l2'),
        skopt.space.Categorical([8, 10, 12], name="batch_size")
    ]

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

    result = tune.run(
        tune.with_parameters(train),
        name="Test SkOpt",
        metric="mean_square_error",
        mode="min",
        search_alg=skopt_search,
        num_samples=10,
        resources_per_trial={"cpu": 1, "gpu": 0}
    )

    best_trial = result.get_best_trial("mean_square_error", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["mean_square_error"]))

    test_best_model(best_trial=best_trial)


if __name__ == "__main__":
    main()
