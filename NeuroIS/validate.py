import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NeuroISDataset


def validate(network, valid_loader):
	with torch.no_grad():
		network.eval()
		criterion = nn.BCELoss()
		correct = 0
		total = 0
		total_loss = 0
		for _ in range(100):
			for graph, num_k, num_n, mask, target in valid_loader:
				graph = graph.cuda()
				num_k = num_k.cuda()
				num_n = num_n.cuda()
				mask = mask.cuda()
				target = target.cuda()
				output = network(graph, num_k, num_n, mask)
				loss = criterion(output, target)
				correct += ((output > 0.5) == target).sum().item()
				total += target.shape[0]
				total_loss += loss.item() * target.shape[0]
		print('Validate Loss: %f  Validate Accuracy: %f%%' % (total_loss / total, correct / total * 100))


if __name__ == '__main__':
	network = torch.load('models/checkpoint_best.pt')
	valid_set = NeuroISDataset('data/valid.data')
	valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False, num_workers=4)
	validate(network, valid_loader)