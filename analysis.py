from ray.tune import Analysis

optuna_analysis = Analysis("/home/knut/ray_results/optuna_skopt_3", default_metric="mean_square_error",
                           default_mode="min")

optuna_df = optuna_analysis.dataframe()
