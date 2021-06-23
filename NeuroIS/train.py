import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from model import NeuroISNet
from dataset import NeuroISDataset
from validate import validate
from time import time


def train(network, train_loader, optimizer, accumulation_steps):
	criterion = nn.BCELoss()
	correct = 0
	total = 0
	steps = 0
	total_loss = 0
	total_loss_sum = 0
	optimizer.zero_grad()
	start_time = time()
	for idx, (graph, num_k, num_n, mask, target) in enumerate(train_loader, 1):
		graph = graph.cuda()
		num_k = num_k.cuda()
		num_n = num_n.cuda()
		mask = mask.cuda()
		target = target.cuda()
		output = network(graph, num_k, num_n, mask)
		loss = criterion(output, target) / accumulation_steps
		total_loss += loss.item()
		total_loss_sum += loss.item() * target.shape[0] * accumulation_steps
		loss.backward()
		steps += 1
		if steps == accumulation_steps:
			clip_grad_norm_(network.parameters(), 0.65)
			optimizer.step()
			end_time = time()
			print('[%d/%d] Loss: %f Time: %f' % (idx, len(train_loader), total_loss, end_time - start_time))
			optimizer.zero_grad()
			start_time = end_time
			steps = 0
			total_loss = 0
		correct += ((output > 0.5) == target).sum().item()
		total += target.shape[0]
	print('Training Loss: %f  Training Accuracy: %f%%' % (total_loss_sum / total, correct / total * 100))
	return total_loss_sum / total, correct / total

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	network = NeuroISNet(128, 26).to("cuda")
	train_set = NeuroISDataset('data/train.data')
	valid_set = NeuroISDataset('data/valid.data')
	train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=4)
	valid_loader = DataLoader(valid_set, batch_size=100, shuffle=False, num_workers=4)
	optimizer = optim.Adam(network.parameters(), 2e-5)
	epoch_lst = []
	train_loss_lst = []
	valid_loss_lst = []
	train_acc_lst = []
	valid_acc_lst = []
	for epoch in range(1, 151):
		print('Training epoch %d...' % epoch)
		train_loss, train_acc = train(network, train_loader, optimizer, 100)
		valid_loss, valid_acc = validate(network, valid_loader)
		torch.save(network, 'models/checkpoint_{}.pt'.format(epoch))

		epoch_lst.append(epoch)
		train_loss_lst.append(train_loss)
		valid_loss_lst.append(valid_loss)
		train_acc_lst.append(train_acc)
		valid_acc_lst.append(valid_acc)

		plt.plot(epoch_lst, train_loss_lst, label='Train')
		plt.plot(epoch_lst, valid_loss_lst, label='Valid')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig('img/loss.png')
		plt.close()
		
		plt.plot(epoch_lst, train_acc_lst, label='Train')
		plt.plot(epoch_lst, valid_acc_lst, label='Valid')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig('img/acc.png')
		plt.close()