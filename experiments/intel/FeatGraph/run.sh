#!/bin/bash



export TVM_HOME=$PWD/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export PYTHONPATH=$PWD/FeatGraph/python:${PYTHONPATH}
export TVM_NUM_THREADS=10
export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";

cd FeatGraph/benchmark
python3.8 download_reddit_dataset.py
python3.8 bench_vanilla_spmm.py --dataset data/reddit_csr_float32.npz --feat-len 512 --target x86

