from common import storage

results_path, results_model_path, gpr_path, acq_path = storage.init_new_results_dir(num_hp=4, search_method="BO")
hpo_current_iteration_dir, final_epoc_dir = storage.init_new_iteration_dir(iteration=0, parent=results_model_path)
current_epoc_dir = storage.init_new_epoc_dir(e=0, parent=hpo_current_iteration_dir)
storage.init_hpo_selection_file(hp=["A", "B", "C"], parent=results_path)
storage.append_hpo_selection_file(hp=["0.4545", "Adam", "Test"], parent=results_path)
storage.append_hpo_selection_file(hp=["0.4545", "Bdam", "Test"], parent=results_path)
storage.append_hpo_selection_file(hp=["0.4545", "Cdam", "Test"], parent=results_path)
storage.append_mse_to_easy_plot(mse=2.45454646, parent=results_path)
storage.append_mse_to_easy_plot(mse=3.45454646, parent=results_path)
storage.append_mse_to_easy_plot(mse=4.45454646, parent=results_path)
storage.append_mse_to_easy_plot(mse=5.45454646, parent=results_path)
