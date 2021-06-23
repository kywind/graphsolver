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
	start = 50
	end = 100
	accs = [50]
	accs_val = [50]
	losses = [0.69]
	losses_val = [0.69]

	train_set = ModifiedNeuroTSPDataset('data/train')
	valid_set = ModifiedNeuroTSPDataset('data/valid')

	used_size = 1000
	train_set, _ = torch.utils.data.random_split(train_set, [used_size, len(train_set) - used_size])
	valid_set, _ = torch.utils.data.random_split(valid_set, [used_size, len(valid_set) - used_size])

	train_loader = DataLoader(train_set, batch_size=30, shuffle=False, num_workers=4)
	valid_loader = DataLoader(valid_set, batch_size=30, shuffle=False, num_workers=4)
	best_acc = (0, 0)
	best_acc_val = (0, 0)

	for t in range(start, end):
		network = torch.load('models/checkpoint_{}.pt'.format(t+1))
		loss, acc = validate(network, train_loader)
		losses.append(loss)
		accs.append(acc)
		if acc > best_acc[0]:
			best_acc = (acc, t)

		loss, acc = validate(network, valid_loader)
		losses_val.append(loss)
		accs_val.append(acc)
		if acc > best_acc_val[0]:
			best_acc_val = (acc, t)

		plt.figure(1)
		plt.plot(accs, label='train_acc')
		plt.plot(accs_val, label='valid_acc')
		plt.legend()
		plt.ylabel("accuracy")
		plt.xlabel("epoch")
		plt.title("accuracy curve")
		# plt.savefig("acc_curve.png")
		plt.close(1)

		plt.figure(1)
		plt.plot(losses, label='train_los')
		plt.plot(losses_val, label='valid_loss')
		plt.ylabel("loss")
		plt.xlabel("epoch")
		plt.title("loss curve")
		# plt.savefig("loss_curve.png")
		plt.close(1)

		print(best_acc, best_acc_val)

		# print(losses, accs)
	