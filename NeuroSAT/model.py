import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

	def __init__(self, in_dim, hidden_dim, out_dim):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(in_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, out_dim)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x


def batched_LSTMCell_forward(cell, x, hc):
	b1, b2, _ = x.shape
	reshaped_x = x.reshape(b1 * b2, -1)
	reshaped_h = hc[0].reshape(b1 * b2, -1)
	reshaped_c = hc[1].reshape(b1 * b2, -1)
	reshaped_out_hc = cell(reshaped_x, (reshaped_h, reshaped_c))
	out_h = reshaped_out_hc[0].reshape(b1, b2, -1)
	out_c = reshaped_out_hc[1].reshape(b1, b2, -1)
	return (out_h, out_c)


class NeuroSATNet(nn.Module):

	def __init__(self, hidden_dim, iters):
		super(NeuroSATNet, self).__init__()
		self.L_init = nn.Linear(1, hidden_dim)
		self.C_init = nn.Linear(1, hidden_dim)
		self.L_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
		self.C_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
		self.L_update = nn.LSTMCell(hidden_dim*2, hidden_dim)
		self.C_update = nn.LSTMCell(hidden_dim, hidden_dim)
		self.L_norm = nn.LayerNorm(hidden_dim)
		self.C_norm = nn.LayerNorm(hidden_dim)
		self.L_vote = MLP(hidden_dim, hidden_dim, 1)
		self.iters = iters
	
	@staticmethod
	def flip(L):
		pos = L[:, 0::2, :]
		neg = L[:, 1::2, :]
		return torch.cat([neg, pos], dim=2).reshape(L.shape)
	
	def forward(self, x, mask):
		bsz, n_var, n_clause = x.shape
		one = torch.ones(1, device=x.device)
		L_init = self.L_init(one).unsqueeze(0).unsqueeze(0).repeat((bsz, n_var, 1))
		C_init = self.C_init(one).unsqueeze(0).unsqueeze(0).repeat((bsz, n_clause, 1))
		L_hc = (L_init, torch.zeros(L_init.shape, device=x.device))
		C_hc = (C_init, torch.zeros(C_init.shape, device=x.device))
		L = self.L_norm(L_hc[0])

		for iter in range(self.iters):
			ML_msg = torch.bmm(x.transpose(1, 2), self.L_msg(L))
			C_hc = batched_LSTMCell_forward(self.C_update, ML_msg, C_hc)
			C = self.C_norm(C_hc[0])
			MC_msg = torch.bmm(x, self.C_msg(C))
			L_hc = batched_LSTMCell_forward(self.L_update, torch.cat([MC_msg, self.flip(L)], dim=2), L_hc)
			L = self.L_norm(L_hc[0])
		
		L_vote = self.L_vote(L).squeeze(2)
		L_mean = (L_vote * mask).sum(dim=1)
		return torch.sigmoid(L_mean)