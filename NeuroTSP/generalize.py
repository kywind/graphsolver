import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ModifiedNeuroTSPDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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
	return total_loss / total, correct / total * 100

if __name__ == '__main__':
	accs = []
	x = [50, 55, 60, 65, 70]
	for n in x:

		valid_set = ModifiedNeuroTSPDataset('data/valid_{}'.format(n))
		used_size = 1000
		valid_set, _ = torch.utils.data.random_split(valid_set, [used_size, len(valid_set) - used_size])
		valid_loader = DataLoader(valid_set, batch_size=30, shuffle=False, num_workers=4)

		network = torch.load('models/checkpoint_58.pt').cuda()


		loss, acc = validate(network, valid_loader)
		accs.append(acc)

	plt.figure(1)
	plt.plot(x, accs)
	# plt.legend()
	plt.ylabel("accuracy")
	plt.xlabel("graph_size")
	plt.title("accuracy curve")
	plt.savefig("acc_gen.png")
	plt.close(1)

	