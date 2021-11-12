import getopt
import sys

import skopt
from ray import tune

from rayTune_bayesOpt import optimize as run_bayesopt
from rayTune_optuna_skOpt import optimize as run_optuna
from rayTune_skOpt import optimize as run_skopt


def main(argv):
    try:
        arguments, values = getopt.getopt(argv[1:], "sbo")
    except getopt.GetoptError:
        sys.exit(2)

    if len(arguments) != 1:
        print("Please select an optimizer")
        print("-s for Scikit-Optimize")
        print("-b for Bayesian Optimization")
        print("-o for Optuna")
        sys.exit(1)

    for current_argument, current_value in arguments:
        if current_argument in "-s":
            print("Running Scikit Optimize")
            run_skopt(
                space=[
                    skopt.space.Integer(2, 5, name="hidden_layers"),
                    skopt.space.Integer(40, 60, name="hidden_layer_width"),
                    skopt.space.Real(10 ** -5, 10 ** 0, "log-uniform", name='lr'),
                    skopt.space.Real(10 ** -3, 10 ** 0, "uniform", name='l2'),
                    skopt.space.Categorical([8, 10, 12], name="batch_size")
                ]
            )
        elif current_argument in "-b":
            print("Running Bayesian Optimization")
            run_bayesopt(
                config={
                    "l2": tune.uniform(1e-3, 1),
                    "lr": tune.loguniform(1e-5, 1),
                    "batch_size": tune.uniform(8, 12),
                    "hidden_layers": tune.quniform(2, 5, 1),
                    "hidden_layer_width": tune.quniform(40, 60, 1)
                }
            )
        elif current_argument in "-o":
            print("Running Optuna")
            run_optuna(
                config={
                    "l2": tune.uniform(1e-3, 1),
                    "lr": tune.loguniform(1e-5, 1),
                    "batch_size": tune.uniform(8, 12),
                    "hidden_layers": tune.quniform(2, 5, 1),
                    "hidden_layer_width": tune.quniform(40, 60, 1)
                }
            )


if __name__ == "__main__":
    main(sys.argv)
