import torch
import random
from torch.utils.data import Dataset


class NeuroISDataset(Dataset):

	def __init__(self, dir):
		super(NeuroISDataset, self).__init__()
		self.graphs = []
		self.num_n = []
		self.num_k = []
		self.masks = []
		n_nodes_list = []
		edges_list = []
		max_n = 0
		file = open(dir, 'r')
		n_data = int(file.readline().rstrip())
		for idx in range(n_data):
			n_nodes_str, _ = file.readline().rstrip().split()
			n_nodes = int(n_nodes_str)
			n_nodes_list.append(n_nodes)
			edges = []
			for node_x in range(n_nodes):
				node_y_list = file.readline().rstrip().split()
				for node_y_str in node_y_list:
					node_y = int(node_y_str) - 1
					edges.append([node_x, node_y])
			edges_list.append(edges)
			ans_list = file.readline().rstrip().split()
			self.num_n.append(torch.tensor(n_nodes))
			self.num_k.append(torch.tensor(len(ans_list)))
			max_n = max(max_n, n_nodes)
			if idx % 1000 == 0:
				print('[%d]' % idx)
		file.close()
		
		for idx in range(n_data):
			indices = torch.tensor(edges_list[idx]).t()
			values = torch.ones(indices.shape[1])
			graph = torch.sparse_coo_tensor(indices, values, (max_n, max_n))
			self.graphs.append(graph)
			n_nodes = n_nodes_list[idx]
			mask = torch.zeros(max_n)
			mask[:n_nodes] = 1 / n_nodes
			self.masks.append(mask)
			if idx % 1000 == 0:
				print('[%d]' % idx)

	def __len__(self):
		return len(self.num_n)
	
	def __getitem__(self, idx):
		p = random.random()
		if p < 0.25:
			generated_k = torch.randint(1, self.num_n[idx].item() + 1, ())
		elif p < 0.5:
			generated_k = random.choice([self.num_k[idx] - 1, self.num_k[idx], self.num_k[idx] + 1, self.num_k[idx] + 2])
		else:
			generated_k = random.choice([self.num_k[idx], self.num_k[idx] + 1])
		target = (generated_k <= self.num_k[idx]).float()
		return self.graphs[idx].to_dense(), generated_k, self.num_n[idx], self.masks[idx], target