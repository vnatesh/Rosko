#!/bin/bash

# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";

# compile taco 
export TACO_PATH=$PWD/taco/build/lib;
export LD_LIBRARY_PATH=$TACO_PATH:$LD_LIBRARY_PATH

g++ -std=c++11 -fopenmp -pthread -O3 -DNDEBUG -DTACO -I taco/include -L$PWD/taco/build/lib taco_spmm.cpp -o taco_spmm -ltaco


# compile mkl_dense
g++ -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.cpp -Wl,--no-as-needed \
-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread \
-lm -ldl -o mkl_sgemm_test


# compile mkl_sparse CSR gemm test with Intel MKL
x=$PWD
sudo cp sparse_gemm.cpp /opt/intel/oneapi/mkl/2021.1.1/examples/sycl/spblas
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/sparse_gemm.out $x
cd $x


# compile rosko_sgemm_test
make;

mkdir -p reports_arm_trans
i=291;
NTRIALS=1;

echo "algo,M,K,N,nz,id,time" >> result_dlmc

for x in 7 8 9 95 98;
do
	for file in dlmc/transformer/magnitude_pruning/0.$x/*.smtx; 
	do

		./rosko_sgemm_test $file $i 100 1; 
		./taco_spmm $file $i $NTRIALS 1; 
		./mkl_sgemm_test $file $i $NTRIALS 1;
		./sparse_gemm.out $file $i $NTRIALS 1;

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_rosko_$i ./rosko_sgemm_test $file $i 1 0;

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_cake_$i ./cake_sgemm_test $file $i 1 0;

		# # perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# # -o reports_arm_trans/report_setup_cake_$i ./rosko_sgemm_test $file $i;

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_armpl_$i ./arm_test $file $i 1 0; 

		# # perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# # -o reports_arm_trans/report_setup_armpl_$i ./arm_test $file $i; 

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_armcl_$i ./neon_sgemm $file $i 1 0;

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_setup_armcl_$i ./neon_sgemm $file $i;

		((i++));

	done
done
# run matmul bench


# python3 plots.py $NTRIALS; 


