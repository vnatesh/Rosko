#!/bin/bash

# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";

x=$PWD
sudo cp mkl_sparse_gemm.cpp /opt/intel/oneapi/mkl/2021.1.1/examples/sycl/spblas
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/mkl_sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/mkl_sparse_gemm.out $x
cd $x