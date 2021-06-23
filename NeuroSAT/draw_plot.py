import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def draw_and_save(record, path=''):
    a = np.array(record)
    names = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc']
    colors = ['b', 'r', 'b', 'r']
    num_epoch = a.shape[0]
    x_major_locator = MultipleLocator((num_epoch+4)//5)
    x = np.arange(num_epoch) + 1

    for i in range(4):
        fig, axis = plt.subplots()
        axis.set_xlabel('Epoch')
        axis.plot(x, a[:, i], color=colors[i], label=names[i])
        axis.xaxis.set_major_locator(x_major_locator)
        axis.set_ylabel(names[i])
        handles, labels = axis.get_legend_handles_labels()
        plt.legend(handles, labels, loc='best')
        plt.savefig(path + names[i] + '.png')
        plt.close()