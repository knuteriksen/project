import os

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from ray.tune import Analysis

from rayTune_common.constants import metric, mode
from rayTune_common.test import test_model
from rayTune_common.utils import config_to_model

data = []
path_to_run_results = "/run_results_test/run6"
path_to_csv = "/run_results_test/run6/results.csv"

list_experiments = [f.path for f in os.scandir(path_to_run_results) if f.is_dir()]
list_experiments.sort(key=lambda x: x.split("_")[-1])

for experient_number, path_to_experiment in enumerate(list_experiments):
    print(f"Experiment: {experient_number} -- {path_to_experiment}")
    experiment_data = {}

    list_experiment_trials = [f.path for f in os.scandir(path_to_experiment) if f.is_dir()]
    list_experiment_trials.sort(key=lambda x: int(x.split("_")[4]))

    best_trial_analysis = Analysis(path_to_experiment, default_metric=metric, default_mode=mode)
    best_trial_config = best_trial_analysis.get_best_config(metric=metric, mode=mode)
    best_trial_logdir = best_trial_analysis.get_best_logdir(metric=metric, mode=mode)
    list_best_trial_checkpoints = [f.path for f in os.scandir(best_trial_logdir) if f.is_dir()]
    list_best_trial_checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    best_trial_checkpoint_path = os.path.join(list_best_trial_checkpoints[-1], "checkpoint")
    best_trial_model = config_to_model(config=best_trial_config, checkpoint_path=best_trial_checkpoint_path)
    best_trial_mse = test_model(model=best_trial_model, batch_size=best_trial_config["batch_size"])
    print(best_trial_mse)

    for trial_number, path_to_trial in enumerate(list_experiment_trials):
        # print(f"Trial_ {trial_number + 1} -- {path_to_trial}")

        list_trial_checkpoints = [f.path for f in os.scandir(path_to_trial) if f.is_dir()]
        list_trial_checkpoints.sort(key=lambda x: int(x.split("_")[-1]))

        trial_checkpoint_path = os.path.join(list_trial_checkpoints[-1], "checkpoint")

        trial_analysis = Analysis(path_to_trial, default_metric=metric, default_mode=mode)
        trial_config = trial_analysis.get_best_config(metric=metric, mode=mode)

        trial_model = config_to_model(config=trial_config, checkpoint_path=trial_checkpoint_path)

        trial_mse = test_model(model=trial_model, batch_size=trial_config["batch_size"])
        experiment_data[trial_number] = trial_mse

    sorted_experiment_data = dict(sorted(experiment_data.items()))
    data.append(sorted_experiment_data)

df = pd.DataFrame(data)
df["best"] = df.min(axis=1)
df.loc["mean"] = df.mean(axis=0)
df.loc["var"] = df.var(axis=0)
print(df)
df.to_csv(path_to_csv)
mean = df.loc["mean"]

for index, row in df.iterrows():
    data = []
    for i in range(100):
        if i == 0:
            data.append(row[i])
        else:
            if row[i] < data[i - 1]:
                data.append(row[i])
            else:
                data.append(data[i - 1])
    sns.lineplot(data=data)
    plt.show()
