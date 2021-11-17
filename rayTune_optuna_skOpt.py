import numpy as np
import torch
from optuna.integration import SkoptSampler
from ray import tune
from ray.tune.suggest.optuna import OptunaSearch

from rayTune_common.constants import random_seed, metric, mode
from rayTune_common.train import train


def optimize(config: {}, iterations: int, experiment_name: str, logdir: str):
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
        metric=metric,
        mode=mode,
    )

    # algo = ConcurrencyLimiter(algo, max_concurrent=1)

    result = tune.run(
        tune.with_parameters(train),
        name=experiment_name,
        config=config,
        metric=metric,
        mode=mode,
        search_alg=algo,
        num_samples=iterations,
        resources_per_trial={"cpu": 8, "gpu": 0},
        verbose=1,
        checkpoint_score_attr="min-mean_square_error",
        keep_checkpoints_num=1,
        local_dir=logdir
    )

    """
    best_trial = get_best_trial(result)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial best mean square error: {}".format(
        best_trial.last_result[metric]))

    best_trial_model = trial_to_model(best_trial)
    test_model(model=best_trial_model, batch_size=best_trial.config["batch_size"])
    """
