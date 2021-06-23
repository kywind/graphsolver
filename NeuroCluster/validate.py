import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ClusterDataset


def validate(network, valid_loader):
	network.eval()
	with torch.no_grad():
		criterion = nn.BCELoss()
		correct = 0
		total = 0
		total_loss = 0
		for idx, (EV, W, R, k, mask, clustering_exists) in enumerate(valid_loader, 1):
			EV, W, R, k, mask, clustering_exists = EV.cuda(), W.cuda(), R.cuda(), k.cuda(), mask.cuda(), clustering_exists.cuda()
			EV = EV.reshape((-1,) + EV.shape[2:])
			W = W.reshape((-1,) + W.shape[2:])
			R = R.reshape((-1,) + R.shape[2:])
			k = k.reshape((-1,) + k.shape[2:])
			mask = mask.reshape((-1,) + mask.shape[2:])
			clustering_exists = clustering_exists.reshape((-1,) + clustering_exists.shape[2:])

			output = network(EV, W, R, k, mask)
			loss = criterion(output, clustering_exists)
			total_loss += loss.item() * clustering_exists.shape[0]
			correct += ((output > 0.5) == clustering_exists).sum().item()
			total += clustering_exists.shape[0]
		validate_loss = total_loss / total
		validate_acc = correct / total * 100
		print('Validate Loss: %f  Validate Accuracy: %f%%' % (validate_loss, validate_acc))
		return(validate_loss, validate_acc)


if __name__ == '__main__':
	network = torch.load('models/checkpoint_best.pt')
	valid_set = ClusterDataset('data/valid.txt')
	valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False, num_workers=4)
	validate(network, valid_loader)