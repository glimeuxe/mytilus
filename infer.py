#= infer.py

import os, shutil, torch
import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from data import *

PROGRESS_COLUMNS = (
	TextColumn("[progress.description]{task.description:<25}"),
	BarColumn(bar_width=40),
	TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
	TimeElapsedColumn(),
)

def evaluate(model, data_loader, gallery_path, thickening_threshold, device, is_dry_run=False):
	model.eval()
	watertight_count = 0
	total_eligible_count = 0
	with torch.no_grad(), Progress(*PROGRESS_COLUMNS) as progress:
		progress_task = progress.add_task("[cyan]Evaluating...", total=len(data_loader))
		for batch in data_loader:
			# select single item from batch using [0] indexing
			vertices = batch["vertices"][0].to(device)
			faces = batch["faces"][0].to(device)
			adjacency = batch["adjacency"][0].to(device)
			center = batch["center"][0].to(device)
			scale = batch["scale"][0].to(device)
			base_name = batch["base_name"][0]
			input_path = batch["input_path"][0]
			output_path_target = batch["output_path"][0]
			target_vertices = batch["target_vertices"][0].to(device)
			target_faces = batch["target_faces"][0].to(device)
			maximum_thickening = batch["maximum_thickening"][0].to(device)
			output_vertices, output_faces, _, _, _ = model(vertices, faces, adjacency, maximum_thickening, use_hard_thickening=True, thickening_threshold=thickening_threshold)

			# check if mesh ought to be watertight if its target is
			if is_watertight(target_faces.cpu()):
				total_eligible_count += 1
				if is_watertight(output_faces.cpu()):
					watertight_count += 1

			# un-normalize vertices for metric calculation and saving
			output_vertices_unnormalized = output_vertices.clone()
			if scale.item() > 1e-8:
				output_vertices_unnormalized = output_vertices_unnormalized * scale
			output_vertices_unnormalized = output_vertices_unnormalized + center

			save_obj(os.path.join(gallery_path, f"{base_name}_pred.obj"), output_vertices_unnormalized.cpu(), output_faces.cpu(), is_dry_run)
			if not is_dry_run:
				shutil.copy(input_path, os.path.join(gallery_path, f"{base_name}_input.obj"))
				shutil.copy(output_path_target, os.path.join(gallery_path, f"{base_name}_target.obj"))
			else:
				print(f"[INFO] Dry run: would copy {input_path} to {os.path.join(gallery_path, f'{base_name}_input.obj')}")
				print(f"[INFO] Dry run: would copy {output_path_target} to {os.path.join(gallery_path, f'{base_name}_target.obj')}")
			progress.update(progress_task, advance=1)
	print(f"\n[INFO] Inference complete, predictions saved to \"{gallery_path}\".")
	watertight_fraction = "NA"
	if total_eligible_count > 0:
		watertight_fraction = watertight_count / total_eligible_count
		print(f"[INFO] Watertight Fraction: {watertight_fraction:.4f} ({watertight_count}/{total_eligible_count} eligible meshes)")
	else:
		print(f"[INFO] Watertight Fraction: NA (0 eligible meshes)")
	return watertight_fraction