#!/bin/bash

# Run matmul bench on raspberry pi 3b for 1..4 cores through linux perf

# compile arm_test with ARMPL
gcc -I/opt/arm/armpl_21.1_gcc-9.3/include -fopenmp  arm_test.c -o test.o  /opt/arm/armpl_21.1_gcc-9.3/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib -lm -o arm_test;

# compile ARMCL sgemm test (NEON)
export ARMCL_PATH=/home/ubuntu/ComputeLibrary;
export LD_LIBRARY_PATH=$ARMCL_PATH/build:$LD_LIBRARY_PATH

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

echo "algo,M,K,N,nz,time" >> result_end_to_end


for dir in matrices/* 
do 
	for file in $dir/* 
	do 
		./rosko_sgemm_test $file; 
		./arm_test $file; 
		./neon_sgemm  $file;
	done 
done


# run matmul bench


# python3 plots.py $NTRIALS; 

