#= aid.py

import hashlib, json, os, random, torch
import numpy as np

def seed_backend(seed_value):
	random.seed(seed_value)
	np.random.seed(seed_value)
	torch.manual_seed(seed_value)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed_value)

def get_run_hash(configuration):
	keys_to_hash = [
		"BINARIZATION_WEIGHT",
		"CLASSIFICATION_WEIGHT",
		"GNN_LAYER_DIMENSIONS",
		"HEAD_LAYER_DIMENSIONS",
		"INVERTED_TETRAHEDRA_WEIGHT",
		"LEARNING_RATE",
		"MAX_DATASET_SIZE",
		"MAX_EPOCHS",
		"MAX_THICKENING",
		"PATIENCE",
		"REGRESSION_WEIGHT",
		"SEED",
		"THICKENING_THRESHOLD",
		"TRAIN_VAL_TEST_RATIO"
	]
	return hashlib.sha256(
		json.dumps(
			{key: configuration[key] for key in sorted(keys_to_hash) if key in configuration},
			sort_keys=True
		).encode()
	).hexdigest()[:16]

def log_run(log_path, run_independent_variables, run_dependent_variables, is_dry_run=False):
	all_runs = []
	if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
		with open(log_path, "r") as file_object:
			try:
				all_runs = json.load(file_object)
				if not isinstance(all_runs, list): all_runs = [all_runs]
			except json.JSONDecodeError:
				print(f"[WARNING] Could not decode {log_path}, starting new log file.")
	final_log = {"RUN_ID": run_independent_variables.pop("RUN_ID")}
	final_log.update(dict(sorted(run_independent_variables.items())))
	final_log.update(dict(sorted(run_dependent_variables.items())))
	all_runs.append(final_log)
	if is_dry_run:
		print(f"[INFO] Dry run: would save run data to {log_path}.")
	else:
		with open(log_path, "w") as file_object:
			json.dump(all_runs, file_object, indent=4)
		print(f"[INFO] Saved run data to {log_path}.")