#= train.py

import glob, matplotlib, os, time, torch, yaml, shutil
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from aid import *
from architecture import *
from data import *
from infer import *

def inverted_tetrahedra_loss(vertices, faces, displaced_vertices):
	if faces.numel() == 0:
		return torch.tensor(0.0, device=vertices.device)
	return torch.mean(torch.relu(-torch.sum((((displaced_vertices[faces[:, 0]] + displaced_vertices[faces[:, 1]] + displaced_vertices[faces[:, 2]]) / 3.0) - vertices[faces[:, 0]]) * torch.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]], dim=1), dim=1)))

def get_file_pairs(configuration):
	print("[INFO] Starting preprocessing: finding mesh pairs...")
	file_pairs = []
	for input_path in sorted(glob.glob(os.path.join(configuration["DATASET_PATH"], "*_input.obj"))):
		if os.path.exists(input_path.replace("_input.obj", "_output.obj")):
			file_pairs.append((input_path, input_path.replace("_input.obj", "_output.obj")))
	if not file_pairs:
		raise FileNotFoundError(f"No valid input/output OBJ pairs found in {configuration['DATASET_PATH']}")
	random.shuffle(file_pairs)
	if configuration["MAX_DATASET_SIZE"] is not None:
		file_pairs = file_pairs[:int(configuration["MAX_DATASET_SIZE"])]
		print(f"[INFO] Limited dataset to {len(file_pairs)} samples.")

	print(f"[INFO] Found {len(file_pairs)} valid pairs.")
	return file_pairs

def create_dataloaders(file_pairs, configuration):
	number_of_training_samples = int(len(file_pairs) * configuration["TRAIN_VAL_TEST_RATIO"][0])
	training_pairs = file_pairs[:number_of_training_samples]
	validation_pairs = file_pairs[number_of_training_samples : number_of_training_samples + int(len(file_pairs) * configuration["TRAIN_VAL_TEST_RATIO"][1])]
	testing_pairs = file_pairs[number_of_training_samples + int(len(file_pairs) * configuration["TRAIN_VAL_TEST_RATIO"][1]):]
	print(f"[INFO] Split dataset into {len(training_pairs)} train, {len(validation_pairs)} val, {len(testing_pairs)} test.")
	return (
		DataLoader(MeshDataset(training_pairs, configuration), batch_size=1, shuffle=True),
		DataLoader(MeshDataset(validation_pairs, configuration), batch_size=1),
		DataLoader(MeshDataset(testing_pairs, configuration), batch_size=1)
	)

def train(configuration, training_loader, validation_loader, run_identifier, device, architecture_parameters, is_dry_run=False):
	run_path = os.path.join(configuration["RUNS_PATH"], run_identifier)
	if not is_dry_run:
		os.makedirs(run_path, exist_ok=True)
	model = NeuralThickeningNet(**architecture_parameters).to(device)
	optimizer = optim.AdamW(model.parameters(), lr=configuration["LEARNING_RATE"])
	scheduler = ReduceLROnPlateau(
		optimizer, "min",
		patience=configuration["PATIENCE"] // 2,
		factor=0.5
	)
	training_history = {"training_loss": [], "validation_loss": [], "learning_rate": []}
	best_validation_loss = float("inf")
	epochs_without_improvement = 0
	final_epoch_number = 0
	best_model_path = os.path.join(run_path, "weights.pt")
	with Progress(
		TextColumn("{task.description}"), BarColumn(bar_width=30),
		TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(),
		TextColumn("Training Loss: {task.fields[training_loss]}"),
		TextColumn("Validation Loss: {task.fields[validation_loss]}"),
		TextColumn("Learning Rate: {task.fields[learning_rate]}"),
		TextColumn("Early Stopping Counter: {task.fields[early_stopping_counter]}"),
	) as progress:
		epoch_progress_task = progress.add_task(
			"[cyan]Training",
			total=configuration["MAX_EPOCHS"],
			training_loss="NA",
			validation_loss="NA",
			learning_rate="NA",
			early_stopping_counter="0"
		)
		for epoch_index in range(configuration["MAX_EPOCHS"]):
			final_epoch_number = epoch_index + 1
			training_history["learning_rate"].append(optimizer.param_groups[0]["lr"])

			model.train()
			epoch_training_loss = 0.0
			for batch in training_loader:
				vertices, faces, adjacency, target_mask, target_magnitudes, maximum_thickening = (
					batch["vertices"][0].to(device), batch["faces"][0].to(device),
					batch["adjacency"][0].to(device), batch["target_thickening_mask"][0].to(device),
					batch["target_magnitudes"][0].to(device), batch["maximum_thickening"][0].to(device)
				)
				optimizer.zero_grad()
				_, _, thickening_probabilities, thickening_magnitudes, vertex_normals = model(vertices, faces, adjacency, maximum_thickening)
				loss = torch.tensor(0.0, device=device)
				if thickening_probabilities.numel() > 0:
					displaced_vertices = model.differentiable_displace_vertices(vertices, faces, vertex_normals, thickening_probabilities, thickening_magnitudes)
					target_mask_boolean = target_mask.squeeze().bool()
					loss = (
						configuration["CLASSIFICATION_WEIGHT"] * nn.functional.binary_cross_entropy(thickening_probabilities, target_mask) +
						(configuration["REGRESSION_WEIGHT"] * nn.functional.mse_loss(thickening_magnitudes[target_mask_boolean], target_magnitudes[target_mask_boolean]) if target_mask_boolean.any() else 0.0) +
						configuration["BINARIZATION_WEIGHT"] * torch.mean(thickening_probabilities * (1 - thickening_probabilities)) +
						configuration["INVERTED_TETRAHEDRA_WEIGHT"] * inverted_tetrahedra_loss(vertices, faces, displaced_vertices)
					)
				if torch.isfinite(loss):
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
					optimizer.step()
					epoch_training_loss += loss.item()
			training_history["training_loss"].append(epoch_training_loss / len(training_loader))

			model.eval()
			epoch_validation_loss = 0.0
			with torch.no_grad():
				for batch in validation_loader:
					vertices, faces, adjacency, target_mask, target_magnitudes, maximum_thickening = (
						batch["vertices"][0].to(device), batch["faces"][0].to(device),
						batch["adjacency"][0].to(device), batch["target_thickening_mask"][0].to(device),
						batch["target_magnitudes"][0].to(device), batch["maximum_thickening"][0].to(device)
					)
					_, _, thickening_probabilities, thickening_magnitudes, vertex_normals = model(vertices, faces, adjacency, maximum_thickening)
					loss = torch.tensor(0.0, device=device)
					if thickening_probabilities.numel() > 0:
						displaced_vertices = model.differentiable_displace_vertices(vertices, faces, vertex_normals, thickening_probabilities, thickening_magnitudes)
						target_mask_boolean = target_mask.squeeze().bool()
						loss = (
							configuration["CLASSIFICATION_WEIGHT"] * nn.functional.binary_cross_entropy(thickening_probabilities, target_mask) +
							(configuration["REGRESSION_WEIGHT"] * nn.functional.mse_loss(thickening_magnitudes[target_mask_boolean], target_magnitudes[target_mask_boolean]) if target_mask_boolean.any() else 0.0) +
							configuration["BINARIZATION_WEIGHT"] * torch.mean(thickening_probabilities * (1 - thickening_probabilities)) +
							configuration["INVERTED_TETRAHEDRA_WEIGHT"] * inverted_tetrahedra_loss(vertices, faces, displaced_vertices)
						)
					epoch_validation_loss += loss.item()
			average_validation_loss = epoch_validation_loss / len(validation_loader)
			training_history["validation_loss"].append(average_validation_loss)
			scheduler.step(average_validation_loss)
			if average_validation_loss < best_validation_loss:
				best_validation_loss = average_validation_loss
				epochs_without_improvement = 0
				if not is_dry_run:
					torch.save(model.state_dict(), best_model_path)
			else:
				epochs_without_improvement += 1
			progress.update(
				epoch_progress_task,
				advance=1,
				description=f"Epoch {epoch_index + 1}/{configuration['MAX_EPOCHS']}",
				training_loss=f"{training_history['training_loss'][-1]:.6f}",
				validation_loss=f"{average_validation_loss:.6f}",
				learning_rate=f"{optimizer.param_groups[0]['lr']:.1E}",
				early_stopping_counter=f"{epochs_without_improvement}/{configuration['PATIENCE']}"
			)
			if epochs_without_improvement >= configuration["PATIENCE"]:
				print(f"\n[INFO] Early stopping triggered after {configuration['PATIENCE']} epochs.")
				break
	plot_path = os.path.join(run_path, "losses.png")
	figure, axis1 = plt.subplots(figsize=(12, 6))
	axis1.plot(training_history["training_loss"], label="Training Loss", color="tab:blue")
	axis1.plot(training_history["validation_loss"], label="Validation Loss", color="tab:orange")
	axis1.set_xlabel("Epoch")
	axis1.set_ylabel("Hybrid Loss")
	axis1.legend(loc="upper left")
	axis1.grid(True)
	axis2 = axis1.twinx()
	axis2.plot(training_history["learning_rate"], label="Learning Rate", color="tab:green", linestyle="--")
	axis2.set_ylabel("Learning Rate", color="tab:green")
	axis2.tick_params(axis="y", labelcolor="tab:green")
	axis2.legend(loc="upper right")
	figure.suptitle(f"Training & Validation Loss (Run: {run_identifier})")
	figure.tight_layout()
	if not is_dry_run:
		plt.savefig(plot_path)
	print(f"[INFO] Saved loss plot to {plot_path}")
	plt.close(figure)
	return best_model_path, final_epoch_number, f"{best_validation_loss:.7f}"

def main():
	with open("config.yaml", "r") as file_object:
		CONFIG = yaml.safe_load(file_object)
	is_dry_run = CONFIG["DRY_RUN"]
	COMPUTATION_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	CONFIG["RUN_ID"] = f"{get_run_hash(CONFIG)}-{int(time.time())}"
	seed_backend(CONFIG["SEED"])
	torch.backends.cudnn.benchmark = True
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	print(f"[INFO] Using {torch.__version__}, {COMPUTATION_DEVICE} | Run ID: {CONFIG['RUN_ID']}.")
	if is_dry_run:
		print("[WARNING] Running in dry run mode. No files will be written.")
	os.makedirs(CONFIG["RUNS_PATH"], exist_ok=True)
	architecture_parameters = {
		"gnn_layer_dimensions": CONFIG["GNN_LAYER_DIMENSIONS"],
		"head_layer_dimensions": CONFIG["HEAD_LAYER_DIMENSIONS"],
	}
	try:
		file_pairs = get_file_pairs(CONFIG)
		if not file_pairs:
			print("[ERROR] No valid file pairs found.")
			return
	except FileNotFoundError as error:
		print(f"[ERROR] {error}")
		return
	training_loader, validation_loader, testing_loader = create_dataloaders(file_pairs, CONFIG)

	training_start_time = time.time()
	best_model_path, number_of_epochs, best_validation_loss_string = train(CONFIG, training_loader, validation_loader, CONFIG["RUN_ID"], COMPUTATION_DEVICE, architecture_parameters, is_dry_run)
	watertight_fraction = "NA"
	if testing_loader and os.path.exists(best_model_path):
		model = NeuralThickeningNet(**architecture_parameters).to(COMPUTATION_DEVICE)
		model.load_state_dict(torch.load(best_model_path, map_location=COMPUTATION_DEVICE))
		run_path = os.path.join(CONFIG["RUNS_PATH"], CONFIG["RUN_ID"])
		watertight_fraction = evaluate(model, testing_loader, run_path, CONFIG["THICKENING_THRESHOLD"], COMPUTATION_DEVICE, is_dry_run)
	else:
		print("\n[WARNING] No test data or model path available, skipping evaluation.")
	run_independent_variables = CONFIG.copy()
	for key in ["DATASET_PATH", "LOG_PATH", "RUNS_PATH"]:
		if key in run_independent_variables:
			del run_independent_variables[key]
	run_independent_variables["ARCHITECTURE"] = str(NeuralThickeningNet(**architecture_parameters)).replace("\n", "").replace(" ", "")

	log_run(
		CONFIG["LOG_PATH"],
		run_independent_variables,
		{
			"best_validation_loss": best_validation_loss_string,
			"number_of_epochs": number_of_epochs,
			"minutes_trained": int((time.time() - training_start_time) / 60),
			"watertight_fraction": f"{watertight_fraction:.4f}" if isinstance(watertight_fraction, float) else watertight_fraction,
		},
		is_dry_run
	)

if __name__ == "__main__":
	main()