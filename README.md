# Hyperparameter Tuning with Ray Tune and PyTorch

The neural network is adapted from [TTK28 at NTNU](https://github.com/bgrimstad/TTK28-Courseware)

## Environment setup (to run locally)

1. Download and install Anaconda (https://www.anaconda.com/).
2. Create a new conda environment: `conda env create -f environment.yml`. This will create a new environment called
   ttk28 with the packages listed in `environment.yml`.
3. Activate the new environment: `conda activate hpo`.

## Running the experiments

The Bayesian Optimization experiment is divided into four notebooks:

- expBo_1.ipynb
- expBo_2.ipynb
- expBo_3.ipynb
- expBo_4.ipynb

This was done to run the experiment simultaneously at 4 different computers, to reduce time.

## rayTune_common

This folder contains five different files:

- `configs.py` - The different configuration spaces
- `constants.py` - Different constants used throughout the notebook
- `model.py` - The neural network adapted to work with Tune
- `test.py` - The function used to calculate test MSE
- `utils.py` - Function for converting Tune config into Net model

## UnseededRun_results

This folder contains the results of the experiment

## Analyzing results

- `analysis.ipynb` - Extracts results, trains on validation set, and evaluates test MSE
- `plots.ipynb` - Used to make plots
- `table.ipynb` - Used to make a csv file to convert into latex table

## Preprocessing

- `data_preperation.py`

## Tensorbard visualization

`tensorboard --logdir=path_to_log_dir`
