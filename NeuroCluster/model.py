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


class NeuroCenterNet(nn.Module):

	def __init__(self, hidden_dim, iters):
		super(NeuroCenterNet, self).__init__()
		self.V_init = nn.Linear(2, hidden_dim)
		self.E_init = nn.Linear(2, hidden_dim)
		self.V_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
		self.E_msg = MLP(hidden_dim, hidden_dim, hidden_dim)
		self.V_update = nn.LSTMCell(hidden_dim, hidden_dim)
		self.E_update = nn.LSTMCell(hidden_dim, hidden_dim)
		self.V_norm = nn.LayerNorm(hidden_dim)
		self.E_norm = nn.LayerNorm(hidden_dim)
		self.V_vote = MLP(hidden_dim, hidden_dim, 1)
		self.iters = iters
	
	def forward(self, EV, W, R, k, mask):
		bsz, n_edge, n_vertex = EV.shape
		# one = torch.ones(1, device=EV.device)
		ones = torch.ones((bsz, n_vertex, 1), device=EV.device)
		V_init = self.V_init(torch.cat((ones, ones * k), dim=2))
		E_init = self.E_init(torch.cat((W, R), dim=2))
		V_hc = (V_init, torch.zeros(V_init.shape, device=EV.device))
		E_hc = (E_init, torch.zeros(E_init.shape, device=EV.device))
		E = self.E_norm(E_hc[0])

		for _ in range(self.iters):
			# print(self.E_msg(E).shape)
			E2V_msg = torch.bmm(EV.transpose(1, 2), self.E_msg(E))
			V_hc = batched_LSTMCell_forward(self.V_update, E2V_msg, V_hc)
			V = self.V_norm(V_hc[0])
			V2E_msg = torch.bmm(EV, self.V_msg(V))
			E_hc = batched_LSTMCell_forward(self.E_update, V2E_msg, E_hc)
			E = self.E_norm(E_hc[0])
		V_vote = self.V_vote(V).squeeze(2)
		V_mean = (V_vote * mask).sum(dim=1)
		return torch.sigmoid(V_mean)