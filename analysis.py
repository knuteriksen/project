import os

from ray.tune import Analysis

from rayTune_common.test import test_model
from rayTune_common.utils import config_to_model

path_to_experiment = "/home/knut/ray_results/b_000"
list_experiment_trials = [f.path for f in os.scandir(path_to_experiment) if f.is_dir()]
list_experiment_trials.sort(key=lambda x: x.split("_")[4])

for i, trial in enumerate(list_experiment_trials):
    list_trial_checkpoints = [f.path for f in os.scandir(trial) if f.is_dir()]
    list_trial_checkpoints.sort(key=lambda x: x.split("_")[-1])
    best_trial_checkpoint_path = os.path.join(list_trial_checkpoints[-1], "checkpoint")

    best_trial_analysis = Analysis(trial, default_metric="mean_square_error", default_mode="min")
    best_trial_config = best_trial_analysis.get_best_config(metric="mean_square_error", mode="min")

    best_trial_model = config_to_model(config=best_trial_config, checkpoint_path=best_trial_checkpoint_path)

    best_trial_test_mse = test_model(model=best_trial_model, batch_size=best_trial_config["batch_size"])
