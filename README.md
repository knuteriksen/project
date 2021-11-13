# Hyperparameter Tuning with Ray Tune, Optuna and PyTorch

[Based on code from TTK28 at NTNU](https://github.com/bgrimstad/TTK28-Courseware)

## Environment setup (to run locally)

1. Download and install Anaconda (https://www.anaconda.com/).
2. Create a new conda environment: `conda env create -f environment.yml`. This will create a new environment called
   ttk28 with the packages listed in `environment.yml`.
3. Activate the new environment: `conda activate ttk28`.

## Running Experiment

`python3 main.py -x 1`
This will run the optimization loop 25 times, with 100 optimization iterations each time

## Tensorbard

`tensorboard --logdir=~/ray_results/Test\ SkOpt/`

## Runnnig Smoke Tests

Checkout commit `57b87f89bb643abdc1965b8273658fb3069eedfd`:`random search added` to run smoke tests

With Scikit Optimize: `python3 main.py -s 10`  
With Bayesian Optimization: `python3 main.py -b 10`  
With Optuna and Scikit Optimize: `python3 main.py -o 10`  
With Random serach: `python3 main.py -r 10`
