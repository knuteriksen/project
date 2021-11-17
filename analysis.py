import os

import pandas as pd
from ray.tune import Analysis

from rayTune_common.constants import logdir_6, metric, mode
from rayTune_common.test import test_model
from rayTune_common.utils import config_to_model

data = []
path_to_run_results = logdir_6
path_to_csv = "/home/knut/results/run6/results.csv"

list_experiments = [f.path for f in os.scandir(path_to_run_results) if f.is_dir()]
list_experiments.sort(key=lambda x: x.split("_")[-1])

for experient_number, path_to_experiment in enumerate(list_experiments):
    print(f"Experiment: {experient_number} -- {path_to_experiment}")
    experiment_data = {}

    list_experiment_trials = [f.path for f in os.scandir(path_to_experiment) if f.is_dir()]
    list_experiment_trials.sort(key=lambda x: x.split("_")[4])

    for trial_number, path_to_trial in enumerate(list_experiment_trials):
        print(f"Trial_ {trial_number} -- {path_to_trial}")
        list_trial_checkpoints = [f.path for f in os.scandir(path_to_trial) if f.is_dir()]
        list_trial_checkpoints.sort(key=lambda x: x.split("_")[-1])
        best_trial_checkpoint_path = os.path.join(list_trial_checkpoints[-1], "checkpoint")

        best_trial_analysis = Analysis(path_to_trial, default_metric=metric, default_mode=mode)
        best_trial_config = best_trial_analysis.get_best_config(metric=metric, mode=mode)

        best_trial_model = config_to_model(config=best_trial_config, checkpoint_path=best_trial_checkpoint_path)

        best_trial_mse = test_model(model=best_trial_model, batch_size=best_trial_config["batch_size"])
        experiment_data[str(trial_number)] = best_trial_mse

    sorted_experiment_data = dict(sorted(experiment_data.items()))
    data.append(sorted_experiment_data)

df = pd.DataFrame(data)
df["best"] = df.min(axis=1)
df.loc["mean"] = df.mean(axis=0)
print(df)
df.to_csv(path_to_csv)
