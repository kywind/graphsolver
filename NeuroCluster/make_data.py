import numpy as np
import subprocess
import argparse
from os import listdir
import time

def read_and_write_graph_buffer(args, idx):
    input_path = args.in_dir + '%d.graph' % idx
    output_path = 'buffer/input_%d.txt' % idx
    input_file = open(input_path, 'r')
    output_file = open(output_path, 'w')
    input_file.readline()
    n = int(input_file.readline().rstrip()[11:])
    for _ in range(n * (n - 1) // 2 + 6):
        input_file.readline()
    dis = np.zeros((n, n))
    for i in range(n):
        s = input_file.readline().rstrip().split()
        for j in range(n):
            dis[i, j] = float(s[j])
            if j < i:
                dis[i, j] = dis[j, i]
    input_file.close()
    k = np.random.randint(2, 6)
    output_file.write('%d %d' % (n, k))
    for i in range(n):
        output_file.write('\n')
        for j in range(n):
            output_file.write('%f ' % dis[i, j])
    output_file.close()
    input_path = 'buffer/input_%d.txt' % idx
    output_path = 'buffer/output_%d.txt' % idx
    input_buffer = open(input_path, 'r')
    output_buffer = open(output_path, 'w')
    subprocess.run('./solver', stdin=input_buffer, stdout=output_buffer)
    

def write_data(file, data_id):
    input_path = 'buffer/input_%d.txt' % data_id
    output_path = 'buffer/output_%d.txt' % data_id
    input_buffer = open(input_path, 'r')
    output_buffer = open(output_path, 'r')
    n_node, k = input_buffer.readline().rstrip().split()
    n_node = int(n_node)
    k = int(k)
    file.write('\n%d %d' % (n_node, k))
    for _ in range(n_node):
        input_str = input_buffer.readline().rstrip()
        file.write('\n%s' % input_str)
    for _ in range(k + 1):
        output_str = output_buffer.readline().rstrip()
        file.write('\n%s' % output_str)
    input_buffer.close()
    output_buffer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', action='store', dest='in_dir', type=str, default='data/cluster_15/')
    parser.add_argument('--out_dir', action='store', dest='out_dir', type=str, default='data/')
    parser.add_argument('--n_train', action='store', dest='n_train', type=int, default=1000000)
    parser.add_argument('--n_valid', action='store', dest='n_valid', type=int, default=100000)
    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=5)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)
    parser.add_argument('--min_r', action='store', dest='min_r', type=float, default=0.2)
    parser.add_argument('--max_r', action='store', dest='max_r', type=float, default=0.5)
    parser.add_argument('--seed', action='store', dest='seed', type=int, default=None)
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    
    subprocess.run(['mkdir', 'buffer/'])

    data_num = len(listdir(args.in_dir))

    for idx in range(data_num):
        st = time.time()
        read_and_write_graph_buffer(args, idx)
        if idx+1 % 1 == 0:
            print('[%d, %.3f]' % (idx, time.time() - st))

    
    # file_train = open(args.out_dir + 'train.txt', 'w')
    # file_valid = open(args.out_dir + 'valid.txt', 'w')
    file_large = open(args.out_dir + 'valid_' + args.in_dir[-3:-1] +'.txt', 'w')
    # pos = data_num * 9 // 10
    # file_train.write('%d' % pos)
    # file_valid.write('%d' % (data_num - pos))
    file_large.write('%d' % data_num)
    for idx in range(data_num):
        # if idx < pos:
        #     write_data(file_train, idx)
        # else:
        #     write_data(file_valid, idx)
        write_data(file_large, idx)
        if idx % 100 == 0:
            print('[%d]' % idx)

    
    # file_train.close()
    # file_valid.close()
    file_large.close()

    subprocess.run(['rm', '-r', 'buffer/'])