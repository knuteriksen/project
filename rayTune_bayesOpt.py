import torch
import torch.utils.data

from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
from common.constants import random_seed

from rayTune_common.training_loop import train
from rayTune_common.test_loop import test_best_model


def optimize(config: {}):
    # Random seed
    torch.manual_seed(random_seed)

    bayesopt = BayesOptSearch(
        random_search_steps=5,
        utility_kwargs={
            "kind": "ei",
            "xi": 0.001,
            "kappa": 2.5
        }
    )

    result = tune.run(
        tune.with_parameters(train),
        name="Test Bayes Opt",
        metric="mean_square_error",
        mode="min",
        search_alg=bayesopt,
        num_samples=10,
        config=config,
        resources_per_trial={"cpu": 1, "gpu": 0}
    )

    best_trial = result.get_best_trial("mean_square_error", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["mean_square_error"]))

    test_best_model(best_trial=best_trial)


if __name__ == "__main__":
    optimize(
        config={
            "l2": tune.uniform(1e-3, 1),
            "lr": tune.loguniform(1e-5, 1),
            "batch_size": tune.uniform(8, 12),
            "hidden_layers": int(tune.uniform(2, 5)),
            "hidden_layer_width": int(tune.uniform(40, 60))
        }
    )
