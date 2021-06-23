import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import NeuroISDataset
from sklearn.decomposition import PCA

def test(network, valid_loader):
	with torch.no_grad():
		criterion = nn.BCELoss()
		network.eval()
		pca_list = []
		for idx, (graph, num_k, num_n, mask, target) in enumerate(valid_loader):
			graph = graph.cuda()
			num_k = num_k.cuda()
			num_n = num_n.cuda()
			mask = mask.cuda()
			target = target.cuda()
			hidden = network.get_hidden(graph, num_k, num_n, mask)
			hidden = hidden.squeeze(0).detach().cpu().numpy()
			output = network(graph, num_k, num_n, mask)
			loss = criterion(output, target)
			nodes = (mask != 0).sum().item()
			pca = PCA(2)
			proj_hidden = pca.fit_transform(hidden)
			pca_list.append(pca.components_[0])
			for i in range(nodes):
				if i + 1 in [1, 7, 9, 13, 14, 18, 28, 29, 36]:
					plt.scatter(proj_hidden[i, 0], proj_hidden[i, 1], color='r')
				else:
					plt.scatter(proj_hidden[i, 0], proj_hidden[i, 1], color='b')
			plt.xlim(-8, 12)
			plt.ylim(-4, 5)
			plt.savefig('img/img_%d_%d_%d_%d_%d.png' % (idx, num_n.item(), num_k.item(), target.item(), (output > 0.5).item()))
			plt.close()
		pca = PCA(2)
		proj_hidden = pca.fit_transform(pca_list)
		for i in range(len(proj_hidden)):
			if i < 5:
				color = 'y'
			elif i < 7:
				color = 'r'
			elif i < 10:
				color = 'b'
			else:
				color = 'g'
			plt.scatter(proj_hidden[i, 0], proj_hidden[i, 1], color=color)
		plt.savefig('img/total0.png')
		plt.close()

if __name__ == '__main__':
	network = torch.load('models/checkpoint_150.pt')
	test_set = NeuroISDataset('data/test.data')
	test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
	test(network, test_loader)