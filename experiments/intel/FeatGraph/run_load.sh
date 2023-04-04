#!/bin/bash


export TVM_HOME=$PWD/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export PYTHONPATH=$PWD/FeatGraph/python:${PYTHONPATH}
# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";


python3.8 write_csr.py
make

# cp download_datasets.py FeatGraph/benchmark
# cd FeatGraph/benchmark
python3.8 download_datasets.py

NTRIALS=5;
NCORES=10
len=512

# cd ../..


echo "algo,file,M,K,N,p,sp,time" >> result_gnn


for (( n=1; n<=$NCORES; n++ ))
do
	./write_rosko $len $n reddit_data reddit_packed
	./write_rosko $len $n ogbn-proteins ogbn_packed

	./rosko_sgemm_test $len $n $NTRIALS reddit_packed 0 0
	./rosko_sgemm_test $len $n $NTRIALS ogbn_packed 0 0

	export TVM_NUM_THREADS=$n

	python3.8 feat_test.py --dataset data/reddit_csr_float32.npz  \
	--feat-len $len --target x86 --nruns $NTRIALS --setup 0 > feat_reddit_$n

	python3.8 feat_test.py --dataset data/ogbn-proteins_csr.npz \
	--feat-len $len --target x86 --nruns $NTRIALS --setup 0 > feat_ogbn_$n 
done


mv result_gnn result_gnn_load


