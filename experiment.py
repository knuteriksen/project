from rayTune_common.configs import config6
from rayTune_common.constants import logdir_6
from rayTune_optuna_skOpt import optimize as run_optuna


def experiment():
    for i in range(3):
        print("Starting New Experiment")
        experiment_name = "experiment_" + str(i).rjust(3, "0")
        run_optuna(
            config=config6,
            iterations=10,
            experiment_name=experiment_name,
            logdir=logdir_6
        )
