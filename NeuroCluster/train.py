import os
from numpy.core.records import record
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from model import NeuroCenterNet
from dataset import ClusterDataset
from time import time
from tqdm import tqdm
import numpy as np
from draw_plot import draw_and_save
import argparse
from validate import validate
# from instance_loader import InstanceLoader
# from validate import validate


def train(network, train_loader, optimizer, accumulation_steps):
	network.train()
	criterion = nn.BCELoss()
	correct = 0
	total = 0
	steps = 0
	total_loss = 0
	running_loss = []
	optimizer.zero_grad()
	tbar = tqdm(train_loader)
	idx = 0
	for EV, W, R, k, mask, clustering_exists in tbar:
		# R for radius, the radius for each cluster
		idx += 1
		EV, W, R, k, mask, clustering_exists = EV.cuda(), W.cuda(), R.cuda(), k.cuda(), mask.cuda(), clustering_exists.cuda()
		# rep_vertices, rep_edges = rep_vertices.cuda(), rep_edges.cuda()
		# EV, W, R, mask, clustering_exists = EV.unsqueeze(0), W.unsqueeze(0), R.unsqueeze(0), mask.unsqueeze(0), clustering_exists.unsqueeze(0)
		EV = EV.reshape((-1,) + EV.shape[2:])
		W = W.reshape((-1,) + W.shape[2:])
		R = R.reshape((-1,) + R.shape[2:])
		k = k.reshape((-1,) + k.shape[2:])
		mask = mask.reshape((-1,) + mask.shape[2:])
		clustering_exists = clustering_exists.reshape((-1,) + clustering_exists.shape[2:])

		output = network(EV, W, R, k, mask)
		loss = criterion(output, clustering_exists) / accumulation_steps
		total_loss += loss.item()
		running_loss.append(loss.item() * accumulation_steps)
		loss.backward()
		steps += 1
		if steps == accumulation_steps:
			clip_grad_norm_(network.parameters(), 0.65)
			optimizer.step()
			tbar.set_description('[%d/%d] Loss: %f' % (idx, len(train_loader), total_loss))
			optimizer.zero_grad()
			steps = 0
			total_loss = 0
		correct += ((output > 0.5) == clustering_exists).sum().item()
		total += clustering_exists.shape[0]
	train_acc = correct / total * 100
	avg_loss = sum(running_loss) / len(running_loss)
	print('Training Accuracy: %f%%, Average Loss: %.5f' % (train_acc, avg_loss))
	return avg_loss, train_acc


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='1')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	network = NeuroCenterNet(128, 26).to("cuda")
	train_set = ClusterDataset('data/train.txt')
	valid_set = ClusterDataset('data/valid.txt')
	train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=4)
	valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False, num_workers=4)

	optimizer = optim.Adam(network.parameters(), 2e-5)
	record_for_curve = []
	for epoch in range(1, 101):
		print('Training epoch %d...' % epoch)
		train_loss, train_acc = train(network, train_loader, optimizer, 10)
		validate_loss, validate_acc = validate(network, valid_loader)
		record_for_curve.append((train_loss, validate_loss, train_acc, validate_acc))
		np.savetxt("record.csv", record_for_curve, delimiter=',')
		draw_and_save(record_for_curve)
		torch.save(network, 'models/checkpoint_{}.pt'.format(epoch))