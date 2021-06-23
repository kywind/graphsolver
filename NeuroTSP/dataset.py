import torch
from torch.utils.data import Dataset
import os
import random
import copy
import numpy as np
from tqdm import tqdm


class ModifiedNeuroTSPDataset(Dataset):

	def __init__(self, path, dev=0.02, training_mode='relational', target_cost=None):
		super(ModifiedNeuroTSPDataset, self).__init__()
		self.dev = dev
		self.training_mode = training_mode
		self.target_cost = target_cost
		self.path = path
		self.filenames = [path + '/' + x for x in os.listdir(path)]
		self.EV_list = []
		self.W_list = []
		self.cost_list = []
		self.mask_list = []
		random.shuffle(self.filenames)

		Ma_list = []
		Mw_list = []
		n_list = []
		m_list = []
		max_n = 0
		max_m = 0
		for idx in tqdm(range(len(self.filenames))):
			Ma, Mw, route = self.read_graph(self.filenames[idx])
			Ma = torch.tensor(Ma)
			Mw = torch.tensor(Mw)
			n = Ma.shape[0]
			m = Ma.nonzero(as_tuple=False).shape[0]
			Ma_list.append(Ma)
			Mw_list.append(Mw)
			n_list.append(n)
			m_list.append(m)
			max_n = max(max_n, n)
			max_m = max(max_m, m)
			cost = sum([Mw[min(x,y), max(x,y)] for (x,y) in zip(route,route[1:]+route[:1])])
			self.cost_list.append(cost)
			# if idx % 1000 == 0:
			# 	print('[%d]' % idx)
		
		for idx in tqdm(range(len(self.filenames))):
			n = n_list[idx]
			m = m_list[idx]
			Ma = Ma_list[idx]
			Mw = Mw_list[idx]
			edges = Ma.nonzero(as_tuple=False)
			W = torch.zeros(max_m, 1)
			W[:m, 0] = Mw[tuple(edges.T)]
			edges = edges.flatten()
			num_edge = torch.arange(m).unsqueeze(1).repeat(1, 2).flatten()
			EV_indices = torch.stack((num_edge, edges))
			EV_values = torch.ones(2 * m)
			EV = torch.sparse_coo_tensor(EV_indices, EV_values, (max_m, max_n))
			mask = torch.zeros(max_m)
			mask[:m] = 1 / m
			self.EV_list.append(EV)
			self.W_list.append(W)
			self.mask_list.append(mask)
			# if idx % 1000 == 0:
			# 	print('[%d]' % idx)
	
	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		EV = self.EV_list[idx].to_dense()
		W = self.W_list[idx]
		mask = self.mask_list[idx]
		cost = self.cost_list[idx]
		doubled_EV = EV.unsqueeze(0).repeat(2, 1, 1)
		doubled_W = W.unsqueeze(0).repeat(2, 1, 1)
		doubled_mask = mask.unsqueeze(0).repeat(2, 1)
		if self.target_cost:
			C = torch.full_like(doubled_W, self.target_cost)
		else:
			C_neg = torch.full_like(doubled_W[0], (1-self.dev)*cost)
			C_pos = torch.full_like(doubled_W[0], (1+self.dev)*cost)
			C = torch.stack((C_neg, C_pos))
		route_exists = (cost <= C[:, 0, 0]).float()
		return doubled_EV, doubled_W, C, doubled_mask, route_exists
	
	@staticmethod
	def read_graph(filepath):
	    with open(filepath,"r") as f:
	        line = ''

	        # Parse number of vertices
	        while 'DIMENSION' not in line: line = f.readline();
	        n = int(line.split()[1])
	        Ma = np.zeros((n,n),dtype=int)
	        Mw = np.zeros((n,n),dtype=float)

	        # Parse edges
	        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
	        line = f.readline()
	        while '-1' not in line:
	            i,j = [int(x) for x in line.split()]
	            Ma[i,j] = 1
	            line = f.readline()

	        # Parse edge weights
	        while 'EDGE_WEIGHT_SECTION' not in line: line = f.readline();
	        for i in range(n):
	            Mw[i,:] = [float(x) for x in f.readline().split()]

	        # Parse tour
	        while 'TOUR_SECTION' not in line: line = f.readline();
	        route = [int(x) for x in f.readline().split()]

	    return Ma,Mw,route


class NeuroTSPDataset(Dataset):

	def __init__(self, path, dev=0.02, training_mode='relational', target_cost=None):
		super(NeuroTSPDataset, self).__init__()
		self.dev = dev
		self.training_mode = training_mode
		self.target_cost = target_cost
		self.path = path
		self.filenames = [path + '/' + x for x in os.listdir(path)]
		random.shuffle(self.filenames)
	
	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		# Read graph from file
		Ma,Mw,route = self.read_graph(self.filenames[idx])
		# Yield two copies of the same instance
		Ma = torch.tensor(Ma)
		Mw = torch.tensor(Mw)
		return Ma,Mw,route

	def collate_fn(self, instances):

		# n_instances: number of instances
		n_instances = len(instances)
		instances_2 = copy.deepcopy(instances)
		instances.extend(instances_2)
        
		# n_vertices[i]: number of vertices in the i-th instance
		n_vertices = torch.tensor([x[0].shape[0] for x in instances])
		# n_edges[i]: number of edges in the i-th instance
		n_edges = torch.tensor([len(torch.nonzero(x[0]).T[0]) for x in instances])
		# total_vertices: total number of vertices among all instances
		total_vertices = sum(n_vertices)
		# total_edges: total number of edges among all instances
		total_edges = sum(n_edges)

		# Compute matrices M, W, CV, CE
		# and vectors edges_mask and route_exists
		EV = torch.zeros((total_edges,total_vertices))
		W = torch.zeros((total_edges,1))
		C = torch.zeros((total_edges,1))
		mask = torch.zeros((n_instances * 2, total_edges))

		# UNSAT + SAT
		route_exists = torch.cat((torch.zeros((n_instances,)), torch.ones((n_instances,))))

		for (i,(Ma,Mw,route)) in enumerate(instances):
			# Get the number of vertices (n) and edges (m) in this graph
			n, m = n_vertices[i], n_edges[i]
			# Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
			n_acc = sum(n_vertices[0:i])
			m_acc = sum(n_edges[0:i])

			mask[i, m_acc:m_acc+m] = 1. / m

			# Get the list of edges in this graph
			edges = torch.nonzero(Ma)

			# Populate EV, W and edges_mask
			for e,(x,y) in enumerate(edges):
				EV[m_acc+e,n_acc+x] = 1
				EV[m_acc+e,n_acc+y] = 1
				W[m_acc+e] = Mw[x,y]

			# Compute the cost of the optimal route
			cost = sum([Mw[min(x,y),max(x,y)] for (x,y) in zip(route,route[1:]+route[:1])]) / n

			if self.target_cost is None:
			    C[m_acc:m_acc+m,0] = (1-self.dev)*cost if i < n_instances else (1+self.dev)*cost
			else:
			    C[m_acc:m_acc+m,0] = self.target_cost

		return EV, W, C, mask, route_exists, n_vertices, n_edges

	@staticmethod
	def read_graph(filepath):
	    with open(filepath,"r") as f:
	        line = ''

	        # Parse number of vertices
	        while 'DIMENSION' not in line: line = f.readline();
	        n = int(line.split()[1])
	        Ma = np.zeros((n,n),dtype=int)
	        Mw = np.zeros((n,n),dtype=float)

	        # Parse edges
	        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
	        line = f.readline()
	        while '-1' not in line:
	            i,j = [int(x) for x in line.split()]
	            Ma[i,j] = 1
	            line = f.readline()

	        # Parse edge weights
	        while 'EDGE_WEIGHT_SECTION' not in line: line = f.readline();
	        for i in range(n):
	            Mw[i,:] = [float(x) for x in f.readline().split()]

	        # Parse tour
	        while 'TOUR_SECTION' not in line: line = f.readline();
	        route = [int(x) for x in f.readline().split()]

	    return Ma,Mw,route