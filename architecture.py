#= architecture.py

import torch
import torch.nn as nn
from collections import defaultdict

class GNNLayer(nn.Module):
	def __init__(self, input_features, output_features):
		super(GNNLayer, self).__init__()
		self.linear = nn.Linear(input_features * 2, output_features)
		self.layer_normalization = nn.LayerNorm(output_features) # fix: add layernorm for stability
		self.activation_function = nn.LeakyReLU(0.2)

	def forward(self, face_features, adjacency):
		if face_features.shape[0] == 0:
			return face_features
		adjacency_mask = (adjacency != -1).float().unsqueeze(-1)
		mean_neighbor_features = (face_features[adjacency.clamp(min=0)] * adjacency_mask).sum(dim=1) / adjacency_mask.sum(dim=1).clamp(min=1)
		return self.activation_function(self.layer_normalization(self.linear(torch.cat([face_features, mean_neighbor_features], dim=1))))

def compute_vertex_normals(vertices, faces):
	if faces.numel() == 0:
		return torch.zeros_like(vertices)
	vertex_normals = torch.zeros_like(vertices)
	vertex0, vertex1, vertex2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]

	# set dim=1 explicitly to resolve userwarning
	face_normals = torch.cross(vertex1 - vertex0, vertex2 - vertex0, dim=1)

	for index in range(3):
		vertex_normals.index_add_(0, faces[:, index], face_normals)
	return nn.functional.normalize(vertex_normals, p=2, dim=1)

class NeuralThickeningNet(nn.Module):
	def __init__(self, gnn_layer_dimensions, head_layer_dimensions):
		super(NeuralThickeningNet, self).__init__()
		initial_features = 8 # define features: center(3), normal(3), area(1), curvature(1)
		layers = []
		input_dimension = initial_features
		for output_dimension in gnn_layer_dimensions:
			layers.append(GNNLayer(input_dimension, output_dimension))
			input_dimension = output_dimension
		self.gnn_stack = nn.ModuleList(layers)

		# double head input dimension to accommodate global context vector
		head_input_dimension = (gnn_layer_dimensions[-1] if gnn_layer_dimensions else initial_features) * 2

		# build prediction heads dynamically based on head_layer_dimensions
		self.classification_head = self._create_head(head_input_dimension, head_layer_dimensions, add_sigmoid_layer=True)
		self.regression_head = self._create_head(head_input_dimension, head_layer_dimensions, add_sigmoid_layer=True)

	def _create_head(self, input_dimension, hidden_dimensions, add_sigmoid_layer=False):
		layers = []
		current_dimension = input_dimension
		for hidden_dimension in hidden_dimensions:
			layers.append(nn.Linear(current_dimension, hidden_dimension))
			layers.append(nn.LeakyReLU(0.2))
			current_dimension = hidden_dimension
		layers.append(nn.Linear(current_dimension, 1))
		if add_sigmoid_layer:
			layers.append(nn.Sigmoid())
		return nn.Sequential(*layers)

	def forward(self, vertices, faces, adjacency, maximum_thickening, use_hard_thickening=False, thickening_threshold=0.5):
		if faces.numel() == 0 or vertices.numel() == 0:
			# return empty tensors for all 5 expected outputs
			return (
				vertices,
				faces,
				torch.empty(0, 1, device=vertices.device),
				torch.empty(0, 1, device=vertices.device),
				torch.empty_like(vertices)
			)
		face_vertices = vertices[faces]
		face_normals = nn.functional.normalize(torch.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0], dim=1) + 1e-12, p=2, dim=1)

		# compute local curvature feature based on dihedral angles with neighbors
		adjacency_mask = (adjacency != -1)
		neighbor_normals = face_normals[adjacency.clamp(min=0)] * adjacency_mask.unsqueeze(-1)
		average_dot_product = (neighbor_normals * face_normals.unsqueeze(1)).sum(dim=-1).sum(dim=1) / adjacency_mask.sum(dim=1).clamp(min=1)

		feature_tensor = torch.cat([
			vertices[faces].mean(dim=1),
			face_normals,
			torch.norm(torch.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0], dim=1), dim=1, keepdim=True) / 2.0,
			(1.0 - average_dot_product).unsqueeze(-1)
		], dim=1)
		for layer in self.gnn_stack:
			feature_tensor = layer(feature_tensor, adjacency)

		# add global context vector to features of each face
		if feature_tensor.shape[0] > 0:
			global_context_vector_expanded = torch.mean(feature_tensor, dim=0, keepdim=True).expand(feature_tensor.shape[0], -1)
			feature_tensor = torch.cat([feature_tensor, global_context_vector_expanded], dim=1)

		thickening_probabilities = self.classification_head(feature_tensor)
		thickening_magnitudes = self.regression_head(feature_tensor) * maximum_thickening
		vertex_normals = compute_vertex_normals(vertices, faces)
		if use_hard_thickening:
			output_vertices, output_faces = self._hard_thicken(vertices, faces, thickening_probabilities, thickening_magnitudes, vertex_normals, thickening_threshold)
		else:
			# During training, the mesh geometry is ignored, so we return the original mesh.
			# The differentiable displacement is calculated separately for the loss function.
			output_vertices, output_faces = vertices, faces
		return output_vertices, output_faces, thickening_probabilities, thickening_magnitudes, vertex_normals

	def differentiable_displace_vertices(self, vertices, faces, vertex_normals, thickening_probabilities, thickening_magnitudes):
		# average face properties to vertices
		vertex_to_face_probabilities = torch.zeros(vertices.shape[0], 1, device=vertices.device)
		vertex_to_face_magnitudes = torch.zeros(vertices.shape[0], 1, device=vertices.device)
		vertex_counts = torch.zeros(vertices.shape[0], 1, device=vertices.device)

		for index in range(3):
			vertex_indices = faces[:, index]
			vertex_to_face_probabilities.index_add_(0, vertex_indices, thickening_probabilities)
			vertex_to_face_magnitudes.index_add_(0, vertex_indices, thickening_magnitudes)
			vertex_counts.index_add_(0, vertex_indices, torch.ones_like(thickening_probabilities))

		vertex_counts = vertex_counts.clamp(min=1)
		vertex_probabilities = vertex_to_face_probabilities / vertex_counts
		vertex_magnitudes = vertex_to_face_magnitudes / vertex_counts

		# compute displaced vertices
		return vertices + (vertex_probabilities * vertex_magnitudes * vertex_normals)

	def _hard_thicken(self, vertices, faces, probabilities, magnitudes, vertex_normals, thickening_threshold):
		faces_to_thicken_mask = (probabilities.squeeze(-1) > thickening_threshold)
		indices_to_thicken = torch.where(faces_to_thicken_mask)[0]
		if len(indices_to_thicken) == 0:
			return vertices, faces
		vertices_to_offset_indices = torch.unique(faces[indices_to_thicken].flatten())
		directed_edge_counts = defaultdict(int)
		for face_index in indices_to_thicken:
			face = faces[face_index].tolist()
			for index in range(3):
				directed_edge_counts[(face[index], face[(index+1)%3])] += 1
		boundary_edges = [edge for edge in directed_edge_counts if directed_edge_counts.get((edge[1], edge[0]), 0) == 0]
		vertex_magnitudes = torch.zeros(vertices.shape[0], 1, device=vertices.device)
		vertex_counts = torch.zeros(vertices.shape[0], 1, device=vertices.device)
		relevant_faces = faces[indices_to_thicken]
		relevant_magnitudes = magnitudes[indices_to_thicken]
		for index in range(3):
			vertex_indices = relevant_faces[:, index]
			vertex_magnitudes.index_add_(0, vertex_indices, relevant_magnitudes)
			vertex_counts.index_add_(0, vertex_indices, torch.ones(len(vertex_indices), 1, device=vertices.device))
		average_vertex_magnitudes = vertex_magnitudes / vertex_counts.clamp(min=1)
		average_vertex_offsets = average_vertex_magnitudes * vertex_normals
		old_number_of_vertices = vertices.shape[0]
		vertex_map = {old_index.item(): new_index for new_index, old_index in enumerate(vertices_to_offset_indices)}
		new_vertices_positions = vertices[vertices_to_offset_indices] + average_vertex_offsets[vertices_to_offset_indices]
		output_vertices = torch.cat([vertices, new_vertices_positions], dim=0)
		kept_faces = faces[~faces_to_thicken_mask]
		cap_faces_old_indices = faces[indices_to_thicken]
		cap_faces_new_indices = cap_faces_old_indices.clone()
		for row_index in range(cap_faces_old_indices.shape[0]):
			for column_index in range(3):
				cap_faces_new_indices[row_index, column_index] = vertex_map[cap_faces_old_indices[row_index, column_index].item()] + old_number_of_vertices
		wall_faces = []
		for vertex1_old_index, vertex2_old_index in boundary_edges:
			vertex1_new_index = vertex_map[vertex1_old_index] + old_number_of_vertices
			vertex2_new_index = vertex_map[vertex2_old_index] + old_number_of_vertices
			wall_faces.append(torch.tensor([vertex1_old_index, vertex2_old_index, vertex2_new_index], device=vertices.device))
			wall_faces.append(torch.tensor([vertex1_old_index, vertex2_new_index, vertex1_new_index], device=vertices.device))
		components = [kept_faces, cap_faces_new_indices]
		if wall_faces:
			components.append(torch.stack(wall_faces))
		return output_vertices, torch.cat([component for component in components if component.numel() > 0], dim=0)