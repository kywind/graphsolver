import math
import torch
from torch.utils.data import DataLoader
from dataset import NeuroSATDataset

def validate(network, valid_loader):
	with torch.no_grad():
		correct = 0
		total = 0
		total_loss = 0
		for graph, mask, target in valid_loader:
			graph = graph.cuda()
			mask = mask.cuda()
			target = target.cuda()
			output = network(graph, mask)
			loss = (math.log(2) * target + (1 - 2 * target) * output).mean()
			# it should be loss = (math.log(2) * (1 - target) + (2 * target - 1) * output).mean() / accumulation_steps
			total_loss += loss.item() * target.shape[0]
			correct += ((output <= 0.28) == target).sum().item()
			total += target.shape[0]
		validate_loss = total_loss / total
		validate_acc = correct / total * 100
		print('Validate Loss: %f  Validate Accuracy: %f%%' % (validate_loss, validate_acc))
		return(validate_loss, validate_acc)


if __name__ == '__main__':
	valid_set = NeuroSATDataset('data/valid.dimacs')
	valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False, num_workers=4)
	network = torch.load('models/checkpoint_best.pt')
	validate(network, valid_loader)