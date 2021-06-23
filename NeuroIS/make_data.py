import numpy as np
import subprocess
import argparse
import re
from os.path import join

def generate_graph_buffer(n_node, n_edge, data_id):
    adj = np.zeros([n_node, n_node], dtype=bool)
    for _ in range(n_edge):
        while True:
            x, y = np.random.randint(n_node, size=2)
            if not (x == y or adj[x, y]):
                adj[x, y] = adj[y, x] = 1
                break
    input_path = 'buffer/input_%d.txt' % data_id
    output_path = 'buffer/output_%d.txt' % data_id
    file_in = open(input_path, 'w')
    file_in.write('%d %d\n' % (n_node, n_edge))
    for i in range(n_node):	
        for j in range(n_node):
            if adj[i, j]:
                file_in.write('%d ' % (j + 1))
        file_in.write('\n')
    file_in.close()
    file_in = open(input_path, 'r')
    file_out = open(output_path, 'w')
    subprocess.run('./IS_runner', stdin=file_in, stdout=file_out)
    

def write_data(file, data_id):
    input_path = 'buffer/input_%d.txt' % data_id
    output_path = 'buffer/output_%d.txt' % data_id
    input_buffer = open(input_path, 'r')
    output_buffer = open(output_path, 'r')
    n_node, n_edge = input_buffer.readline().rstrip().split()
    n_node = int(n_node)
    n_edge = int(n_edge)
    file.write('%d %d\n' % (n_node, n_edge))
    for _ in range(n_node):
        input_str = input_buffer.readline().rstrip()
        file.write('%s\n' % input_str)
    output_str = output_buffer.readline().rstrip()
    file.write('%s\n' % output_str)
    input_buffer.close()
    output_buffer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', action='store', dest='out_dir', type=str, default='data')
    parser.add_argument('--n_train', action='store', dest='n_train', type=int, default=1000)
    parser.add_argument('--n_valid', action='store', dest='n_valid', type=int, default=0)
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

    for data_id in range(args.n_train + args.n_valid):
        if data_id < args.n_train:
            n_node = np.random.randint(args.min_n, args.max_n + 1)
        else:
            n_node = args.max_n
        total_edge = n_node * (n_node + 1) / 2
        n_edge = int(np.random.uniform(args.min_r, args.max_r) * total_edge)
        generate_graph_buffer(n_node, n_edge, data_id)
        if data_id % args.print_interval == 0:
            print('[%d]' % data_id)
    
    train_file = open(join(args.out_dir, 'train.data'), 'w')
    valid_file = open(join(args.out_dir, 'valid.data'), 'w')
    train_file.write('%d\n' % args.n_train)
    valid_file.write('%d\n' % args.n_valid)

    for data_id in range(args.n_train):
        write_data(train_file, data_id)
    for data_id in range(args.n_valid):
        write_data(valid_file, data_id + args.n_train)
    
    train_file.close()
    valid_file.close()

    subprocess.run(['rm', '-r', 'buffer/'])