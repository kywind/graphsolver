import torch
import os
import random
import copy
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class ClusterDataset(Dataset):

	def __init__(self, path, dev=0.02, training_mode='relational', target_cost=None):
		super(ClusterDataset, self).__init__()
		self.dev = dev
		self.training_mode = training_mode
		self.target_cost = target_cost
		self.path = path
		self.EV_list = []
		self.W_list = []
		self.k_list = []
		self.ans_list = []
		self.mask_list = []

		Ma_list = []
		Mw_list = []
		n_list = []
		m_list = []
		max_n = 0
		max_m = 0

		file = open(self.path, 'r')
		num_data = int(file.readline())

		for idx in tqdm(range(num_data)):
			k, Ma, Mw, ans = self.read_graph(file)
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
			self.k_list.append(torch.tensor(k))
			self.ans_list.append(torch.tensor(ans))
			# if idx % 1000 == 0:
			# 	print('[%d]' % idx)
		
		for idx in tqdm(range(num_data)):
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
			mask = torch.zeros(max_n)
			mask[:n] = 1 / n
			self.EV_list.append(EV)
			self.W_list.append(W)
			self.mask_list.append(mask)
			# if idx % 1000 == 0:
			# 	print('[%d]' % idx)
	
	def __len__(self):
		return len(self.ans_list)

	def __getitem__(self, idx):
		EV = self.EV_list[idx].to_dense()
		W = self.W_list[idx]
		k = self.k_list[idx]
		mask = self.mask_list[idx]
		ans = self.ans_list[idx]
		doubled_EV = EV.unsqueeze(0).repeat(2, 1, 1)
		doubled_W = W.unsqueeze(0).repeat(2, 1, 1)
		doubled_k = k.unsqueeze(0).repeat(2, 1, 1)
		doubled_mask = mask.unsqueeze(0).repeat(2, 1)
		if self.target_cost:
			R = torch.full_like(doubled_W, self.target_cost)
		else:
			R_neg = torch.full_like(doubled_W[0], (1-self.dev)*ans)
			R_pos = torch.full_like(doubled_W[0], (1+self.dev)*ans)
			R = torch.stack((R_neg, R_pos))
		ans_exists = (ans <= R[:, 0, 0]).float()
		return doubled_EV, doubled_W, R, doubled_k, doubled_mask, ans_exists
	
	@staticmethod
	def read_graph(file):
		n, k = file.readline().split()
		n = int(n)
		k = int(k)
		Ma = np.zeros((n, n))
		Mw = np.zeros((n, n))
		for i in range(n):
			s = file.readline().rstrip().split()
			for j in range(n):
				if i != j:
					Ma[i, j] = 1
					Mw[i, j] = float(s[j])
		ans = float(file.readline())
		for _ in range(k):
			file.readline()
		return k, Ma, Mw, ans