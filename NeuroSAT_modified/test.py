from matplotlib import colors
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import NeuroSATDataset
from sklearn.decomposition import PCA

def test(network, valid_loader):
	with torch.no_grad():
		for idx, (graph, mask, target) in enumerate(valid_loader):
			graph = graph.cuda()
			mask = mask.cuda()
			L, L_prob = network.get_hidden(graph, mask)
			L_prob = L_prob.squeeze(0)
			hidden = L.squeeze(0).detach().cpu().numpy()
			nodes = (mask != 0).sum().item()
			pca = PCA(2)
			proj_hidden = pca.fit_transform(hidden)
			for i in range(nodes):
				if L_prob[i, 0] <= L_prob[i, 1]:
					plt.scatter(proj_hidden[i << 1, 0], proj_hidden[i << 1, 1], color='b')
					plt.scatter(proj_hidden[i << 1 | 1, 0], proj_hidden[i << 1 | 1, 1], color='r')
				else:
					plt.scatter(proj_hidden[i << 1, 0], proj_hidden[i << 1, 1], color='r')
					plt.scatter(proj_hidden[i << 1 | 1, 0], proj_hidden[i << 1 | 1, 1], color='b')
			plt.savefig('img/img%d_%d.png' % (idx, target.item()))
			plt.close()



if __name__ == '__main__':
	test_set = NeuroSATDataset('data/test.dimacs')
	test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
	network = torch.load('models/checkpoint_small_80.pt')
	test(network, test_loader)