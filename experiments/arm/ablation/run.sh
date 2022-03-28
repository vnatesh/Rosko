#!/bin/bash

mkdir reports_arm_ablate;

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


# compile inner product, rosko, and cake
make;


for s in 70 72 75 77 80 82 85 87 90 92 95 97 99
do

	# armcl dense MM
	perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
	-o reports_arm_ablate/report_armcl_$i ./neon_sgemm $file $i;

	# CAKE dense MM
	perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
	-o reports_arm_ablate/report_cake_$i ./cake_sgemm_test $file $i;

	# # rosko without density-based reordering
	# perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
	# -o reports_arm_ablate/report_rop_$i ./rop_sgemm_test $file $i;

	# rosko 
	perf stat -e l2d_cache_refill_rd,l2d_cache_refill_wr \
	-o reports_arm_ablate/report_rop_$i ./rop_sgemm_test $file $i;

done


# python plots.py $NTRIALS;
