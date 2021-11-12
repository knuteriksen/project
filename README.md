# Hyperparameter Tuning with ray tune and pyTorch
[Based on code from TTK28 at NTNU](https://github.com/bgrimstad/TTK28-Courseware)

## Environment setup (to run locally)
1. Download and install Anaconda (https://www.anaconda.com/).
2. Create a new conda environment: `conda env create -f environment.yml`. 
This will create a new environment called ttk28 with the packages listed in `environment.yml`. 
3. Activate the new environment: `conda activate ttk28`.

## Runnnig optimization
With Scikit Optimize: `python3 main.py -s`  
With Bayesian Optimization: `python3 main.py -b`  
With Optuna and Scikit Optimize: `python3 main.py -o`
