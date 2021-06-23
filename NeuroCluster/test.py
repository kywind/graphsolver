import torch
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):

    def __init__(self):
        super(TestDataset, self).__init__()
        self.data = ['a','b','c','d','e']
        self.label = [1, 2, 3, 4, 5]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return [(self.data[index], self.label[index]), (self.data[index], self.label[index])]

    def collate_fn(self, samples):
        print(samples)
        # print('hehe')
        # return samples


dataset = TestDataset()
dataloader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=dataset.collate_fn)

for data in dataloader:
    # print(data)
    pass