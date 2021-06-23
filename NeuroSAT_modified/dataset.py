import torch
from torch.utils.data import Dataset


class NeuroSATDataset(Dataset):

	def __init__(self, dir):
		super(NeuroSATDataset, self).__init__()
		self.graphs = []
		self.masks = []
		self.targets = []

		graph_shape = []
		edges_list = []

		file = open(dir, 'r')
		n_data = int(file.readline().rstrip())

		max_var = 0
		max_clause = 0

		for idx in range(1, n_data + 1):
			n_var, n_clause = file.readline().rstrip().split()
			n_var = int(n_var)
			n_clause = int(n_clause)

			graph_shape.append((n_var, n_clause))

			max_var = max(max_var, n_var)
			max_clause = max(max_clause, n_clause)

			edges = []
			for clause in range(n_clause):
				var_strs = file.readline().rstrip().split()
				for var_str in var_strs:
					var = int(var_str)
					if var > 0:
						edges.append([(var - 1) << 1, clause])
					else:
						edges.append([(-var - 1) << 1 | 1, clause])
			
			edges_list.append(edges)
			target = torch.tensor(float(file.readline().rstrip()))
			self.targets.append(target)
			if idx % 1000 == 0:
				print('[%d]' % idx)

		for idx in range(n_data):
			n_var, n_clause = graph_shape[idx]
			mask = torch.zeros(max_var)
			mask[:n_var] = 1 / n_var
			self.masks.append(mask)
			indices = torch.tensor(edges_list[idx]).t()
			values = torch.ones(indices.shape[1])
			graph = torch.sparse_coo_tensor(indices, values, (2 * max_var, max_clause))
			self.graphs.append(graph)
			if idx % 1000 == 0:
				print('[%d]' % idx)
		
	def __len__(self):
		return len(self.targets)
	
	def __getitem__(self, idx):
		return self.graphs[idx].to_dense(), self.masks[idx], self.targets[idx]