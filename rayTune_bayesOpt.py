import numpy as np
import torch
import torch.utils.data
from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch

from rayTune_common.constants import random_seed
from rayTune_common.test import test_model
from rayTune_common.train import train
from rayTune_common.utils import get_best_trial, trial_to_model


def optimize(config: {}, iterations: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    bayesopt = BayesOptSearch(
        metric="mean_square_error",
        mode="min",
        random_search_steps=5,
        utility_kwargs={
            "kind": "ei",
            "xi": 0.05,
            "kappa": 2.5
        },
        verbose=2
    )

    algo = ConcurrencyLimiter(bayesopt, max_concurrent=1)

    result = tune.run(
        tune.with_parameters(train),
        name="bayes_opt",
        metric="mean_square_error",
        mode="min",
        search_alg=algo,
        num_samples=iterations,
        config=config,
        resources_per_trial={"cpu": 1, "gpu": 0},
        verbose=3
    )

    best_trial = get_best_trial(result)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial best mean square error: {}".format(
        best_trial.last_result["mean_square_error"]))

    best_trial_model = trial_to_model(best_trial)
    test_model(model=best_trial_model, batch_size=best_trial.config["batch_size"])
