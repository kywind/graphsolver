import numpy as np
import os
import random
import argparse
import PyMiniSolvers.minisolvers as minisolvers

def write_dimacs_to(file, n_vars, iclauses, target):
    file.write("\n%d %d\n" % (n_vars, len(iclauses)))
    for c in iclauses:
        for x in c:
            file.write("%d " % x)
        file.write("\n")
    file.write("%d" % target)

def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [v + 1 if random.random() < 0.5 else -(v + 1) for v in vs]

def gen_iclause_pair(opts):
    n = random.randint(opts.min_n, opts.max_n)

    solver = minisolvers.MinisatSolver()
    for i in range(n): solver.new_var(dvar=True)

    iclauses = []

    while True:
        k_base = 1 if random.random() < opts.p_k_2 else 2
        k = k_base + np.random.geometric(opts.p_geo)
        iclause = generate_k_iclause(n, k)

        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            iclauses.append(iclause)
        else:
            break

    iclause_unsat = iclause
    iclause_sat = [- iclause_unsat[0] ] + iclause_unsat[1:]
    return n, iclauses, iclause_unsat, iclause_sat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', action='store', dest='out_dir', type=str, default='data')
    parser.add_argument('--n_pairs_train', action='store', dest='n_pairs_train', type=int, default=500000)
    parser.add_argument('--n_pairs_valid', action='store', dest='n_pairs_valid', type=int, default=50000)
    parser.add_argument('--min_n', action='store', dest='min_n', type=int, default=10)
    parser.add_argument('--max_n', action='store', dest='max_n', type=int, default=40)
    parser.add_argument('--p_k_2', action='store', dest='p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', action='store', dest='p_geo', type=float, default=0.4)
    parser.add_argument('--seed', action='store', dest='seed', type=int, default=None)
    parser.add_argument('--print_interval', action='store', dest='print_interval', type=int, default=100)
    opts = parser.parse_args()

    if opts.seed is not None:
        np.random.seed(opts.np_seed)
    
    train_path = os.path.join(opts.out_dir, 'train.dimacs')
    valid_path = os.path.join(opts.out_dir, 'valid.dimacs')
    train_file = open(train_path, 'w')
    valid_file = open(valid_path, 'w')

    train_file.write('%d' % (2 * opts.n_pairs_train))
    for pair in range(opts.n_pairs_train):
        if pair % opts.print_interval == 0:
            print("Train [%d]" % pair)
        n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(opts)

        iclauses.append(iclause_unsat)
        write_dimacs_to(train_file, n_vars, iclauses, 0)

        iclauses[-1] = iclause_sat
        write_dimacs_to(train_file, n_vars, iclauses, 1)

    opts.min_n = opts.max_n
    valid_file.write('%d' % (2 * opts.n_pairs_valid))
    for pair in range(opts.n_pairs_valid):
        if pair % opts.print_interval == 0:
            print("Valid [%d]" % pair)
        n_vars, iclauses, iclause_unsat, iclause_sat = gen_iclause_pair(opts)

        iclauses.append(iclause_unsat)
        write_dimacs_to(valid_file, n_vars, iclauses, 0)

        iclauses[-1] = iclause_sat
        write_dimacs_to(valid_file, n_vars, iclauses, 1)