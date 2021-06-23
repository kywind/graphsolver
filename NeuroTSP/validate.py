import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ModifiedNeuroTSPDataset
from tqdm import tqdm


def validate(network, valid_loader):
	with torch.no_grad():
		criterion = nn.BCELoss()
		correct = 0
		total = 0
		total_loss = 0
		for EV, W, C, mask, route_exists in tqdm(valid_loader):
			EV, W, C, mask, route_exists = EV.cuda(), W.cuda(), C.cuda(), mask.cuda(), route_exists.cuda()
			EV = EV.reshape((-1,) + EV.shape[2:])
			W = W.reshape((-1,) + W.shape[2:])
			C = C.reshape((-1,) + C.shape[2:])
			mask = mask.reshape((-1,) + mask.shape[2:])
			route_exists = route_exists.reshape((-1,) + route_exists.shape[2:])

			output = network(EV, W, C, mask)
			loss = criterion(output, route_exists)
			total_loss += loss.item() * route_exists.shape[0]
			correct += ((output > 0.5) == route_exists).sum().item()
			total += route_exists.shape[0]
		print('Validate Loss: %f  Validate Accuracy: %f%%' % (total_loss / total, correct / total * 100))


if __name__ == '__main__':
	network = torch.load('models_kn/checkpoint_5.pt')
	valid_set = ModifiedNeuroTSPDataset('data/temp')
	valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False, num_workers=4)
	validate(network, valid_loader)