#!/bin/bash

# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";

x=$PWD
sudo cp mkl_sparse_gemm.cpp /opt/intel/oneapi/mkl/2021.1.1/examples/sycl/spblas
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/mkl_sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/mkl_sparse_gemm.out $x
cd $x



#!/bin/bash




make;

echo "algo,p,M,K,N,sp,time" >> results



for xx in 70 75 80 85 90 95 98 99;
do
	./rosko_sgemm_test 40 4 288 2208 192 $xx
done





