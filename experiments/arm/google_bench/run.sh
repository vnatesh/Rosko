#!/bin/bash

# Run matmul bench on raspberry pi 3b for 1..4 cores through linux perf

# compile taco 
export TACO_PATH=$PWD/taco/build/lib;
export LD_LIBRARY_PATH=$TACO_PATH:$LD_LIBRARY_PATH

g++ -std=c++11 -fopenmp -pthread -O3 -DNDEBUG -DTACO -I taco/include -L$PWD/taco/build/lib taco_spmm.cpp -o taco_spmm -ltaco


# compile arm_test with ARMPL
# gcc -I/opt/arm/armpl_21.1_gcc-9.3/include -fopenmp  arm_test.c -o test.o \
#  /opt/arm/armpl_21.1_gcc-9.3/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib \
#  -lm -o arm_test;

gcc -I/opt/arm/armpl_21.0_gcc-10.2/include -fopenmp  arm_test.c -o test.o \
  /opt/arm/armpl_21.0_gcc-10.2/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib \
  -lm -o arm_test;


# compile ARMCL sgemm test (NEON)
export ARMCL_PATH=/home/ubuntu/ComputeLibrary;
export LD_LIBRARY_PATH=$ARMCL_PATH/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

aarch64-linux-gnu-g++ -o neon_sgemm.o -c -Wall -DARCH_ARM -Wextra -pedantic \
-Wdisabled-optimization -Wformat=2 -Winit-self -Wstrict-overflow=2 -Wswitch-default \
-std=c++14 -Woverloaded-virtual -Wformat-security -Wctor-dtor-privacy -Wsign-promo \
-Weffc++ -Wno-overlength-strings -Wlogical-op -Wnoexcept -Wstrict-null-sentinel -C \
-fopenmp -march=armv8-a -DENABLE_NEON -DARM_COMPUTE_ENABLE_NEON -Wno-ignored-attributes \
-DENABLE_FP16_KERNELS -DENABLE_FP32_KERNELS -DENABLE_QASYMM8_KERNELS \
-DENABLE_QASYMM8_SIGNED_KERNELS -DENABLE_QSYMM16_KERNELS -DENABLE_INTEGER_KERNELS \
-DENABLE_NHWC_KERNELS -DENABLE_NCHW_KERNELS -O3 -D_GLIBCXX_USE_NANOSLEEP \
-DARM_COMPUTE_CPP_SCHEDULER=1 -DARM_COMPUTE_OPENMP_SCHEDULER=1 \
-DARM_COMPUTE_GRAPH_ENABLED -DARM_COMPUTE_CPU_ENABLED \
-I$ARMCL_PATH/include -I$ARMCL_PATH -I$ARMCL_PATH neon_sgemm.cpp

aarch64-linux-gnu-g++ -o neon_sgemm -fopenmp \
neon_sgemm.o $ARMCL_PATH/build/utils/Utils.o -L$ARMCL_PATH/build \
-L$ARMCL_PATH -lpthread -larm_compute -larm_compute_core

# compile rosko_sgemm_test
make;

mkdir reports_arm_trans
i=0;
NTRIALS=10;
substr="modality"

echo "algo,M,K,N,nz,id,time" >> result_dlmc


for x in 7 8 9 95 98;
do
	for file in dlmc/transformer/magnitude_pruning/0.$x/*.smtx; 
	do

		if [[ $file == *"$substr"* ]];
		then
		    echo "skipping large file."
    		((i++));
		    continue
		else
		    echo "String does not contain substring."
		fi

		./rosko_sgemm_test $file $i $NTRIALS 1 0; 
		./arm_test $file $i $NTRIALS 1 0; 
		./neon_sgemm  $file $i $NTRIALS 1 0;
		./taco_spmm $file $i $NTRIALS 1 0;  
		rm rand.mtx

		perf stat -e l2d_cache_refill \
		-o reports_arm_trans/report_rosko_$i ./rosko_sgemm_test $file $i $NTRIALS 0 1;

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_cake_$i ./cake_sgemm_test $file $i 1 0;

		perf stat -e l2d_cache_refill \
		-o reports_arm_trans/report_armpl_$i ./arm_test $file $i $NTRIALS 0 1; 

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_setup_armpl_$i ./arm_test $file $i; 

		perf stat -e l2d_cache_refill \
		-o reports_arm_trans/report_armcl_$i ./neon_sgemm $file $i $NTRIALS 0 1;

		# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		# -o reports_arm_trans/report_setup_armcl_$i ./neon_sgemm $file $i;

		perf stat -e l2d_cache_refill \
		-o reports_arm_trans/report_taco_$i ./taco_spmm $file $i $NTRIALS 0 1;
		rm rand.mtx

		perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
		-o reports_arm_trans/report_setup_taco_$i ./taco_spmm $file $i $NTRIALS 0 0;;
		rm rand.mtx

		((i++));
	done
done
# run matmul bench

mv result_dlmc result_dlmc_arm

# python3 plots.py $NTRIALS; 

