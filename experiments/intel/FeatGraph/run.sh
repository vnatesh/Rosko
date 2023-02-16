#!/bin/bash



export TVM_HOME=$PWD/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export PYTHONPATH=$PWD/FeatGraph/python:${PYTHONPATH}
export TVM_NUM_THREADS=10
export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";


python3.8 write_csr.py
make

./cake_sgemm_test 512 10 10 reddit_data
./cake_sgemm_test 512 10 10 ogbn-proteins

cp download_ogbn_dataset.py FeatGraph/benchmark
cd FeatGraph/benchmark
python3.8 download_reddit_dataset.py
python3.8 download_ogbn_dataset.py
python3.8 bench_vanilla_spmm.py --dataset data/reddit_csr_float32.npz --feat-len 512 --target x86
python3.8 bench_vanilla_spmm.py --dataset data/ogbn-proteins_csr.npz --feat-len 512 --target x86

