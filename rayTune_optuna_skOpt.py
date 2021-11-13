import numpy as np
import torch
from optuna.integration import SkoptSampler
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch

from rayTune_common.constants import random_seed
from rayTune_common.test_loop import test_best_model
from rayTune_common.training_loop import train


def optimize(config: {}, iterations: int, experiment_name: str):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    sampler = SkoptSampler(
        skopt_kwargs={
            "base_estimator": "GP",
            "n_initial_points": 5,
            "acq_func": "EI",
            "acq_func_kwargs": {"xi": 0.05}
        }
    )

    algo = OptunaSearch(
        sampler=sampler,
        metric="mean_square_error",
        mode="min",
    )

    # algo = ConcurrencyLimiter(algo, max_concurrent=1)

    result = tune.run(
        tune.with_parameters(train),
        name=experiment_name,
        config=config,
        metric="mean_square_error",
        mode="min",
        search_alg=algo,
        num_samples=iterations,
        resources_per_trial={"cpu": 8, "gpu": 0},
        verbose=1,
        checkpoint_score_attr="min-mean_square_error",
        keep_checkpoints_num=0,
    )

    best_trial = result.get_best_trial("mean_square_error", "min", "all")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial best mean square error: {}".format(
        best_trial.last_result["mean_square_error"]))

    test_best_model(best_trial=best_trial)
