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


class NeuroISNet(nn.Module):

	def __init__(self, hidden_dim, iters):
		super(NeuroISNet, self).__init__()
		self.init = MLP(2, hidden_dim, hidden_dim)
		self.msg = MLP(hidden_dim, hidden_dim, hidden_dim)
		self.update = nn.LSTMCell(hidden_dim, hidden_dim)
		self.norm = nn.LayerNorm(hidden_dim)
		self.vote = MLP(hidden_dim, hidden_dim, 1)
		self.iters = iters
	
	def get_hidden(self, x, k, n, mask):
		_, n_node, _ = x.shape
		nk_info = torch.stack((k.float(), n.float())).T
		init = self.init(nk_info).unsqueeze(1).repeat((1, n_node, 1))
		hc = (init, torch.zeros(init.shape, device=x.device))
		embed = self.norm(hc[0])

		for iter in range(self.iters):
			msg = torch.bmm(x, self.msg(embed))
			hc = batched_LSTMCell_forward(self.update, msg, hc)
			embed = self.norm(hc[0])
		
		return embed
	
	def forward(self, x, k, n, mask):
		embed = self.get_hidden(x, k, n, mask)
		vote = self.vote(embed).squeeze(2)
		mean = (vote * mask).sum(dim=1)
		return torch.sigmoid(mean)