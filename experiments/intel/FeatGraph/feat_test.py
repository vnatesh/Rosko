import scipy
import scipy.sparse
import numpy as np
import argparse
import tvm
from tvm import te
from tvm.topi.util import get_const_tuple
from featgraph.module import VanillaSpMMx86, VanillaSpMMcuda
import sys


def exp_range(start, end, mul):
    while start <= end:
        yield start
        start *= mul


def bench_vanilla_spmm_x86(adj_scipy_csr, feat_len, num_col_partitions, 
    num_feat_partitions, num_runs = 5, setup = 1):
    num_rows = adj_scipy_csr.shape[0]
    num_cols = adj_scipy_csr.shape[1]
    vanilla_spmm_module = VanillaSpMMx86(adj_scipy_csr, num_col_partitions)
    SrcFeat = te.placeholder((num_cols, feat_len))
    input_placeholders = [SrcFeat]
    compute_args = {'num_feat_partitions': num_feat_partitions}
    schedule_args = {}
    vanilla_spmm_module.build(input_placeholders, compute_args, schedule_args)
    src_feat_np = np.random.random(get_const_tuple(SrcFeat.shape)).astype('float32')
    src_feat_tvm = tvm.nd.array(src_feat_np, vanilla_spmm_module.ctx)
    input_tvm_ndarrays = [src_feat_tvm]
    if not setup:
        tcost = vanilla_spmm_module.measure_average_time(input_tvm_ndarrays, num_runs)
        print("average time of {} runs: {} sec".format(num_runs, tcost))


if __name__ == '__main__':
   
    feat_lens = [32,64,128,256,512]
    opt_reddit = [(1, 1, 0.2798449166), (8, 1, 0.44859744160000004), (8, 1, 0.8921740286000001), (8, 2, 1.7827022442), (8, 4, 3.5610106794)]
    opt_ogbn = [(1, 1, 0.1136616622), (1, 1, 0.23875559380000003), (1, 2, 0.5176771682), (1, 4, 1.0367150468), (1, 8, 2.0751339349999998)]
    opt_reddit = dict(zip(feat_lens, opt_reddit))
    opt_ogbn = dict(zip(feat_lens, opt_ogbn))

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="The adjacency matrix in csr format stored as a scipy npz file")
    parser.add_argument("--feat-len", type=int, default=128, help="The feature length")
    parser.add_argument("--target", type=str, default='x86', choices=['x86', 'cuda'])
    parser.add_argument("--nruns", type=int, default=5)
    parser.add_argument("--setup", type=int, default=1)

    args = parser.parse_args()
    adj_scipy_csr = scipy.sparse.load_npz(args.dataset)
    assert adj_scipy_csr.format == 'csr'


    if 'reddit' in args.dataset:
        num_col_partitions, num_feat_partitions = opt_reddit[args.feat_len][0], opt_reddit[args.feat_len][1]
    else:
        num_col_partitions, num_feat_partitions = opt_ogbn[args.feat_len][0], opt_ogbn[args.feat_len][1]       

    if args.target == 'x86':
        bench_vanilla_spmm_x86(adj_scipy_csr, args.feat_len, 
            num_col_partitions, num_feat_partitions, args.nruns, args.setup)
