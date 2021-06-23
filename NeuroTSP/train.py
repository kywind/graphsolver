import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from model import NeuroTSPNet
from dataset import ModifiedNeuroTSPDataset
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
# from instance_loader import InstanceLoader
# from validate import validate


def train(network, train_loader, optimizer, accumulation_steps):
	criterion = nn.BCELoss()
	correct = 0
	total = 0
	steps = 0
	total_loss = 0
	losses = []
	optimizer.zero_grad()
	tbar = tqdm(train_loader)
	for idx, (EV, W, C, mask, route_exists) in enumerate(tbar):

		EV, W, C, mask, route_exists = EV.cuda(), W.cuda(), C.cuda(), mask.cuda(), route_exists.cuda()
		# rep_vertices, rep_edges = rep_vertices.cuda(), rep_edges.cuda()
		# EV, W, C, mask, route_exists = EV.unsqueeze(0), W.unsqueeze(0), C.unsqueeze(0), mask.unsqueeze(0), route_exists.unsqueeze(0)
		EV = EV.reshape((-1,) + EV.shape[2:])
		W = W.reshape((-1,) + W.shape[2:])
		C = C.reshape((-1,) + C.shape[2:])
		mask = mask.reshape((-1,) + mask.shape[2:])
		route_exists = route_exists.reshape((-1,) + route_exists.shape[2:])

		output = network(EV, W, C, mask)
		loss = criterion(output, route_exists)
		total_loss += loss.item() / accumulation_steps
		losses.append(loss.item())
		tbar.set_description("Loss: {}".format(loss.item()))
		loss.backward()
		steps += 1

		if steps == accumulation_steps:
			clip_grad_norm_(network.parameters(), 0.65)
			optimizer.step()
			optimizer.zero_grad()
			steps = 0
			total_loss = 0
			
		correct += ((output > 0.5) == route_exists).sum().item()
		total += route_exists.shape[0]
	print('Training Accuracy: %f%%' % (correct / total * 100))
	
	return  np.mean(losses), correct / total * 100


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	# network = NeuroTSPNet(128, 50).to("cuda")
	network = torch.load('models/checkpoint_72.pt').to("cuda")
	train_set = ModifiedNeuroTSPDataset('data/train')
	train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=4)
	# valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False, num_workers=4)
	optimizer = optim.Adam(network.parameters(), 3e-4)

	# losses = [0.69]
	# accs = [50]
	losses, accs = [], []

	for epoch in range(73, 200):
		print('Training epoch %d...' % epoch)
		loss, acc = train(network, train_loader, optimizer, 10)
		# validate(network, valid_loader)

		losses.append(loss)
		accs.append(acc)

		plt.figure(1)
		plt.plot(losses)
		plt.ylabel("loss")
		plt.xlabel("epoch")
		plt.title("training loss curve")
		plt.savefig("train_loss_curve_final.png")
		plt.close(1)

		plt.figure(1)
		plt.plot(accs)
		plt.ylabel("accuracy")
		plt.xlabel("epoch")
		plt.title("training accuracy curve")
		plt.savefig("train_acc_curve_final.png")
		plt.close(1)

		print(losses)
		print(accs)

		torch.save(network, 'models_final/checkpoint_{}.pt'.format(epoch))