import getopt
import sys

import skopt
from ray import tune

from rayTune_bayesOpt import optimize as run_bayesopt
from rayTune_optuna_skOpt import optimize as run_optuna
from rayTune_skOpt import optimize as run_skopt


def main(argv):
    try:
        arguments, values = getopt.getopt(argv[1:], "s:b:o:")
    except getopt.GetoptError:
        print("GetOptError")
        print("Please select an optimizer and specify iterations")
        print("-s for Scikit-Optimize")
        print("-b for Bayesian Optimization")
        print("-o for Optuna")
        print("Example: python3 main.py -b 100")
        sys.exit(2)

    if len(arguments) != 1:
        print("Invalid specification")
        print("Please select an optimizer and specify iterations")
        print("-s for Scikit-Optimize")
        print("-b for Bayesian Optimization")
        print("-o for Optuna")
        print("Example: python3 main.py -b 100")
        sys.exit(1)

    for current_argument, current_value in arguments:
        its = int(current_value)

        if current_argument in "-s":
            print(f"Running Scikit Optimize with {its} iterations")
            run_skopt(
                space=[
                    skopt.space.Integer(2, 5, name="hidden_layers"),
                    skopt.space.Integer(40, 60, name="hidden_layer_width"),
                    skopt.space.Real(10 ** -5, 10 ** 0, "log-uniform", name='lr'),
                    skopt.space.Real(10 ** -3, 10 ** 0, "uniform", name='l2'),
                    skopt.space.Categorical([8, 10, 12], name="batch_size")
                ],
                iterations=its
            )
        elif current_argument in "-b":
            print(f"Running Bayesian Optimization with {its} iterations")
            run_bayesopt(
                config={
                    "l2": tune.uniform(1e-3, 1),
                    "lr": tune.loguniform(1e-5, 1),
                    "batch_size": tune.uniform(8, 12),
                    "hidden_layers": tune.quniform(2, 5, 1),
                    "hidden_layer_width": tune.quniform(40, 60, 1)
                },
                iterations=its
            )
        elif current_argument in "-o":
            print(f"Running Optuna with {its} iterations")
            run_optuna(
                config={
                    "l2": tune.uniform(1e-3, 1),
                    "lr": tune.loguniform(1e-5, 1),
                    "batch_size": tune.uniform(8, 12),
                    "hidden_layers": tune.quniform(2, 5, 1),
                    "hidden_layer_width": tune.quniform(40, 60, 1)
                },
                iterations=its
            )
        else:
            print("Invalid optimizer")
            print("Please select an optimizer")
            print("-s for Scikit-Optimize")
            print("-b for Bayesian Optimization")
            print("-o for Optuna")
            print("Example: python3 main.py -b 100")


if __name__ == "__main__":
    main(sys.argv)
