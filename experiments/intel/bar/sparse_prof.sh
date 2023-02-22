#!/bin/bash

# cd ../../../CAKE_on_CPU;
# source env.sh;
# cd ../experiments/intel/bar;
mkdir reports;

x=$PWD

# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";

# Download ML_Graph SuiteSparse matrices


# compile mkl_sparse gemm test with Intel MKL
sudo cp sparse_gemm.cpp /opt/intel/oneapi/mkl/2021.1.1/examples/sycl/spblas
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/sparse_gemm.out $x
cd $x


# compile rosko_test
make;

NTRIALS=10;
NCORES=10;

echo "algo,file,M,K,N,p,sparsity,time" >> result_sp
# run matmul bench through intel vtune 


for ((n=10; n <= $NCORES; n++));
do
	for file in data/*.mtx; 
	do
		vtune --collect memory-access -data-limit=0 \
			-result-dir=$PWD/prof_result \
		 	$PWD/sparse_gemm.out $file $n 0 $NTRIALS 1; 
		vtune -report summary -r prof_result -format csv \
			-report-output reports/report_mkl_${file##*/}-$n.csv -csv-delimiter comma;
		rm -rf prof_result;


		vtune --collect memory-access -data-limit=0 \
			-result-dir=$PWD/rosko_result \
			 $PWD/rosko_sgemm_test $file $n 0 $NTRIALS 1 ;
		vtune -report summary -r rosko_result -format csv \
			-report-output reports/report_rosko_${file##*/}-$n.csv -csv-delimiter comma;
		rm -rf rosko_result;

		./sparse_gemm.out $file $n 1 $NTRIALS 0;
		./rosko_sgemm_test $file $n 1 $NTRIALS 0; 
	done
done

# python plots.py $NTRIALS;

