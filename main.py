import getopt
import sys

import skopt
from ray import tune

from experiment import experiment
from rayTune_bayesOpt import optimize as run_bayesopt
from rayTune_optuna_skOpt import optimize as run_optuna
from rayTune_random import optimize as run_random
from rayTune_skOpt import optimize as run_skopt


def main(argv):
    try:
        arguments, values = getopt.getopt(argv[1:], "x:s:b:o:r:")
    except getopt.GetoptError:
        print("GetOptError")
        print("Please select an optimizer and specify iterations")
        print("-s for Scikit-Optimize")
        print("-b for Bayesian Optimization")
        print("-o for Optuna")
        print("-r for Random")
        print("-x for Experiment Script")
        print("Example: python3 main.py -b 100")
        sys.exit(2)

    if len(arguments) != 1:
        print("Invalid specification")
        print("Please select an optimizer and specify iterations")
        print("-s for Scikit-Optimize")
        print("-b for Bayesian Optimization")
        print("-o for Optuna")
        print("-r for Random")
        print("-x for Experiment Script")
        print("Example: python3 main.py -b 100")
        sys.exit(1)

    config = {
        "l2": tune.uniform(1e-3, 1),
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.quniform(8, 12, 1),
        "hidden_layers": tune.quniform(2, 5, 1),
        "hidden_layer_width": tune.quniform(30, 60, 5),
        "dropout": tune.choice([0, 1])
    }

    space = [
        skopt.space.Integer(2, 5, name="hidden_layers"),
        skopt.space.Integer(40, 60, name="hidden_layer_width"),
        skopt.space.Real(10 ** -5, 10 ** 0, "log-uniform", name='lr'),
        skopt.space.Real(10 ** -3, 10 ** 0, "uniform", name='l2'),
        skopt.space.Categorical([8, 10, 12], name="batch_size"),
        skopt.space.Categorical([0, 1], name="dropout")
    ]

    for current_argument, current_value in arguments:
        its = int(current_value)

        if current_argument in "-s":
            print(f"Running Scikit Optimize with {its} iterations")
            run_skopt(space=space, iterations=its)

        elif current_argument in "-b":
            print(f"Running Bayesian Optimization with {its} iterations")
            run_bayesopt(config=config, iterations=its)

        elif current_argument in "-o":
            print(f"Running Optuna with {its} iterations")
            run_optuna(config=config, iterations=its, experiment_name="optuna_smoke_test")

        elif current_argument in "-r":
            print(f"Running Random with {its} iterations")
            run_random(config=config, iterations=its)

        elif current_argument in "-x":
            experiment()

        else:
            print("Invalid optimizer")
            print("Please select an optimizer")
            print("-s for Scikit-Optimize")
            print("-b for Bayesian Optimization")
            print("-o for Optuna")
            print("-r for Random")
            print("-x for Experiment Script")
            print("Example: python3 main.py -b 100")


if __name__ == "__main__":
    main(sys.argv)
