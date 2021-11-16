from rayTune_common.configs import config6
from rayTune_optuna_skOpt import optimize as run_optuna


def experiment():
    for i in range(2):
        print("Starting New Experiment")
        experiment_name = "dddd_" + str(i).rjust(3, "0")
        run_optuna(config=config6, iterations=2, experiment_name=experiment_name)
