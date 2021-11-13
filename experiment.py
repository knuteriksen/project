from rayTune_common.configs import config7
from rayTune_optuna_skOpt import optimize as run_optuna


def experiment():
    for i in range(25):
        print("Starting New Experiment")
        experiment_name = "7_" + str(i).rjust(3, "0")
        run_optuna(config=config7, iterations=100, experiment_name=experiment_name)