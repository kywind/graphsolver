import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from model import NeuroSATNet
from dataset import NeuroSATDataset
from validate import validate
from time import time
from tqdm import tqdm
import argparse
import numpy as np
from draw_plot import draw_and_save


def train(network, train_loader, optimizer, accumulation_steps):
	correct = 0
	total = 0
	steps = 0
	total_loss = 0
	running_loss = []
	optimizer.zero_grad()
	tbar = tqdm(train_loader)
	idx = 0
	for graph, mask, target in tbar:
		graph = graph.cuda()
		mask = mask.cuda()
		target = target.cuda()
		idx += 1
		output = network(graph, mask)
		loss = (math.log(2) * (1 - target) + (2 * target - 1) * output).mean() / accumulation_steps
		# out: sum of entropy for variables
		total_loss += loss.item()
		running_loss.append(loss.item() * accumulation_steps)
		loss.backward()
		steps += 1
		if steps == accumulation_steps:
			clip_grad_norm_(network.parameters(), 0.65)
			optimizer.step()
			optimizer.zero_grad()
			steps = 0
			tbar.set_description('[%d/%d] Loss: %f' % (idx, len(train_loader), total_loss))
			total_loss = 0
		correct += ((output <= 0.28) == target).sum().item()
		total += target.shape[0]
	train_acc = correct / total * 100
	avg_loss = sum(running_loss) / len(running_loss)
	print('Training Accuracy: %f%%, Average Loss: %.5f' % (train_acc, avg_loss))
	return avg_loss, train_acc


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--small_data', action='store_true', default=False)
	parser.add_argument('--gpu', default='1')
	args = parser.parse_args()

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	network = NeuroSATNet(128, 26).to("cuda")
	if args.small_data:
		train_set = NeuroSATDataset('small_data/train.dimacs')
		valid_set = NeuroSATDataset('small_data/valid.dimacs')
	else:
		train_set = NeuroSATDataset('data/train.dimacs')
		valid_set = NeuroSATDataset('data/valid.dimacs')
	train_loader = DataLoader(train_set, batch_size=10, shuffle=True, num_workers=4)
	valid_loader = DataLoader(valid_set, batch_size=10, shuffle=False, num_workers=4)
	optimizer = optim.Adam(network.parameters(), 2e-5)
	record = []
	for epoch in range(1, 101):
		print('Training epoch %d...' % epoch)
		train_loss, train_acc = train(network, train_loader, optimizer, 10)
		validate_loss, validate_acc = validate(network, valid_loader)
		record.append((train_loss, validate_loss, train_acc, validate_acc))
		if args.small_data:
			np.savetxt("record_small.csv", record, delimiter=',')
			draw_and_save(record, 'small_')
			torch.save(network, 'models/checkpoint_small_{}.pt'.format(epoch))
		else:
			np.savetxt("record.csv", record, delimiter=',')
			draw_and_save(record, 'standard_')
			torch.save(network, 'models/checkpoint_{}.pt'.format(epoch))