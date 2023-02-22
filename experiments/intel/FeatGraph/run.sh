#!/bin/bash



export TVM_HOME=$PWD/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export PYTHONPATH=$PWD/FeatGraph/python:${PYTHONPATH}
export TVM_NUM_THREADS=10
export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";


python3.8 write_csr.py
make

cp download_ogbn_dataset.py FeatGraph/benchmark
cd FeatGraph/benchmark
python3.8 download_reddit_dataset.py
python3.8 download_ogbn_dataset.py

NTRIALS=10;
NCORES=10

for len in 32 64 128 256 512;
do
	python3.8 bench_vanilla_spmm.py --dataset data/reddit_csr_float32.npz --feat-len $len --target x86
	python3.8 bench_vanilla_spmm.py --dataset data/ogbn-proteins_csr.npz --feat-len $len --target x86

	./rosko_sgemm_test $len $NCORES $NTRIALS reddit_data 0
	./rosko_sgemm_test $len $NCORES $NTRIALS ogbn-proteins 0


	./write_rosko $len $NCORES reddit_data
	./write_rosko $len $NCORES ogbn-proteins



	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/prof_result \
	 	$PWD/sparse_gemm.out $file $n 0 $NTRIALS 1; 
	vtune -report summary -r prof_result -format csv \
		-report-output reports/report_mkl_${file##*/}-$n.csv -csv-delimiter comma;
	rm -rf prof_result;


	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/rosko_sgemm_test $len $NCORES $NTRIALS reddit_data 1 ;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_rosko_${file##*/}-$n.csv -csv-delimiter comma;
	rm -rf rosko_result;


done
