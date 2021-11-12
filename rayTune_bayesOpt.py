import torch
import torch.utils.data
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

from rayTune_common.constants import random_seed
from rayTune_common.test_loop import test_best_model
from rayTune_common.training_loop import train


def optimize(config: {}):
    torch.manual_seed(random_seed)

    bayesopt = BayesOptSearch(
        random_search_steps=5,
        utility_kwargs={
            "kind": "ei",
            "xi": 0.001,
            "kappa": 2.5
        }
    )

    algo = ConcurrencyLimiter(bayesopt, max_concurrent=1)

    result = tune.run(
        tune.with_parameters(train),
        name="Test Bayes Opt",
        metric="mean_square_error",
        mode="min",
        search_alg=algo,
        num_samples=10,
        config=config,
        resources_per_trial={"cpu": 1, "gpu": 0},
        verbose=3
    )

    best_trial = result.get_best_trial("mean_square_error", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["mean_square_error"]))

    test_best_model(best_trial=best_trial)
