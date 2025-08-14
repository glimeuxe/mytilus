#= data.py

import os, scipy.spatial, torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

def load_obj(file_path):
	vertices, faces = [], []
	with open(file_path, "r") as file_object:
		for line in file_object:
			if line.startswith("v "):
				vertices.append([float(value) for value in line.strip().split()[1:4]])
			elif line.startswith("f "):
				faces.append([vertex - 1 for vertex in [int(vertex_index_string.split("/")[0]) for vertex_index_string in line.strip().split()[1:4]]])
	return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.long)

def save_obj(file_path, vertices, faces, is_dry_run=False):
	if is_dry_run:
		print(f"[INFO] Dry run: would save mesh to {file_path}.")
		return
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	with open(file_path, "w") as file_object:
		for vertex in vertices:
			file_object.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
		for face in faces:
			if len(face) > 0:
				file_object.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def get_face_adjacency(faces):
	edge_to_face_map = defaultdict(list)
	for face_index, face in enumerate(faces):
		for index_in_face in range(3):
			edge_to_face_map[tuple(sorted((face.tolist()[index_in_face], face.tolist()[(index_in_face + 1) % 3])))].append(face_index)
	adjacency_list = [[] for _ in range(len(faces))]
	for edge, face_indices in edge_to_face_map.items():
		if len(face_indices) == 2:
			face_index1, face_index2 = face_indices
			adjacency_list[face_index1].append(face_index2)
			adjacency_list[face_index2].append(face_index1)
	adjacency_padded = torch.full((len(faces), max(len(neighbors) for neighbors in adjacency_list) if adjacency_list else 0), -1, dtype=torch.long)
	for face_index, neighbors in enumerate(adjacency_list):
		if neighbors:
			adjacency_padded[face_index, :len(neighbors)] = torch.tensor(neighbors, dtype=torch.long)
	return adjacency_padded

def is_watertight(faces):
	if faces.numel() == 0:
		return True
	edge_counts = defaultdict(int)
	for face in faces.tolist():
		for index in range(3):
			edge_counts[tuple(sorted((face[index], face[(index + 1) % 3])))] += 1
	for count in edge_counts.values():
		if count != 2:
			return False
	return True

class MeshDataset(Dataset):
	def __init__(self, file_pairs, configuration):
		self.file_pairs = file_pairs
		self.configuration = configuration

	def __len__(self):
		return len(self.file_pairs)

	def __getitem__(self, index):
		input_path, output_path = self.file_pairs[index]
		vertices, faces = load_obj(input_path)
		target_vertices, target_faces = load_obj(output_path)

		# initialize mask once, then perform efficient calculation
		ground_truth_thickening_mask = torch.zeros(len(faces), 1, dtype=torch.float32)
		target_faces_set = {tuple(sorted(face_tuple.tolist())) for face_tuple in target_faces}
		ground_truth_thickening_indices = [i for i, f in enumerate(faces) if tuple(sorted(f.tolist())) not in target_faces_set]
		if ground_truth_thickening_indices:
			ground_truth_thickening_mask[ground_truth_thickening_indices] = 1.0

		# center vertices first
		center = vertices.mean(dim=0)
		vertices = vertices - center
		target_vertices = target_vertices - center

		# compute ground-truth magnitudes for regression target
		# measure distance from each input vertex to nearest point on target surface
		ground_truth_magnitudes = torch.zeros(len(faces), 1, dtype=torch.float32)
		if vertices.numel() > 0 and target_vertices.numel() > 0 and faces.numel() > 0:
			vertex_distances = torch.tensor(scipy.spatial.KDTree(target_vertices.numpy()).query(vertices.numpy(), k=1)[0], dtype=torch.float32).unsqueeze(1)
			# average distances of face's vertices to get per-face ground truth magnitude
			ground_truth_magnitudes = torch.mean(vertex_distances[faces], dim=1)

		# compute average edge length on centered, but not yet scaled, mesh
		average_edge_length = torch.tensor(0.0, dtype=torch.float32)
		if faces.numel() > 0:
			vertex0, vertex1, vertex2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
			average_edge_length = torch.cat([
				torch.norm(vertex1 - vertex0, dim=1),
				torch.norm(vertex2 - vertex1, dim=1),
				torch.norm(vertex0 - vertex2, dim=1)
			]).mean()

		# determine normalization scale and calculate per-mesh max thickening
		scale = torch.max(torch.norm(vertices, dim=1))
		per_mesh_maximum_thickening = torch.tensor(self.configuration["MAX_THICKENING"], dtype=torch.float32)
		if scale.item() > 1e-8:
			per_mesh_maximum_thickening = self.configuration["MAX_THICKENING"] * (average_edge_length / scale)
			vertices = vertices / scale
			target_vertices = target_vertices / scale
			ground_truth_magnitudes = ground_truth_magnitudes / scale

		return {
			"verts": vertices,
			"faces": faces,
			"adjacency": get_face_adjacency(faces),
			"target_verts": target_vertices,
			"target_faces": target_faces,
			"input_path": input_path,
			"output_path": output_path,
			"base_name": os.path.basename(input_path).replace("_input.obj", ""),
			"center": center,
			"scale": scale,
			"gt_thicken_mask": ground_truth_thickening_mask,
			"gt_magnitudes": ground_truth_magnitudes,
			"max_thickening": per_mesh_maximum_thickening
		}