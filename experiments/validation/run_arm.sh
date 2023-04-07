#!/bin/bash


# compile arm_test with ARMPL
gcc -I/opt/arm/armpl_21.0_gcc-10.2/include -fopenmp  arm_test.c -o test.o \
  /opt/arm/armpl_21.0_gcc-10.2/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib \
  -lm -o arm_test;

make;
mkdir reports;

echo "layer,algo,M,K,N,p,sp,time" >> results

NCORES=4;
NTRIALS=10;

tail -n +2 layers.csv > mat_tmp.csv

INPUT=mat_tmp.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
i=0


for sp in 80 85 90 95 98
do
	while read layer Cout Cin Wout Wf Win M K N
	do
		# ./rosko_sgemm_test $M $K $N $NCORES $sp $layer $NTRIALS 0

		# perf stat -e l2d_cache_refill,l1d_cache_refill \
		# -o reports_arm_training/report_rosko_$layer-$sp ./rosko_sgemm_test $M $K $N $NCORES $sp $layer $NTRIALS 1;

		./rosko_io_pred $M $K $N $NCORES $sp $layer
		./write_sp_pack $M $K $N $NCORES $sp 1 1 1 $layer orig
		./rosko_setup $M $K $N $NCORES $sp 1 1 1 $layer $NTRIALS 0 orig
		./rosko_sgemm_test $M $K $N $NCORES $sp 1 1 1 $layer $NTRIALS 0 orig
		./arm_test $M $K $N $NCORES $sp 1 1 1 $layer $NTRIALS 0 orig


		perf stat -e l2d_cache_refill,l2d_cache,bus_access_rd,bus_access_wr,ext_mem_req \
		-o reports/report_setup_$layer-$sp ./rosko_setup $M $K $N $NCORES $sp 1 1 1 $layer $NTRIALS 1 orig;

		perf stat -e l2d_cache_refill,l2d_cache,bus_access_rd,bus_access_wr,ext_mem_req \
		-o reports/report_rosko_$layer-$sp ./rosko_sgemm_test $M $K $N $NCORES $sp 1 1 1 $layer $NTRIALS 1 orig;

		perf stat -e l2d_cache_refill,l2d_cache,bus_access_rd,bus_access_wr,ext_mem_req \
		-o reports/report_armpl_$layer-$sp ./arm_test $M $K $N $NCORES $sp 1 1 1 $layer $NTRIALS 1 orig;


		rm $layer;

	done < $INPUT
done

rm mat_tmp.csv
mv results results_arm_train; 
mv reports reports_arm_train;




mkdir reports;

echo "layer,algo,M,K,N,p,sp,time" >> results

for sp in 80 85 90 95 98
do
	for p in 1 2 3 4
	do
		# ./rosko_sgemm_test 10000 10000 10000 $p $sp 1 1 1 hey $NTRIALS 0

		./rosko_io_pred 2000 2000 2000 $p $sp pack
		./write_sp_pack 2000 2000 2000 $p $sp 1 1 1 pack orig
		./rosko_setup 2000 2000 2000 $p $sp 1 1 1 pack $NTRIALS 0 orig
		./rosko_sgemm_test 2000 2000 2000 $p $sp 1 1 1 pack $NTRIALS 0 orig
		./arm_test 2000 2000 2000 $p $sp 1 1 1 pack $NTRIALS 0 orig


		perf stat -e l2d_cache_refill,l2d_cache,bus_access_rd,bus_access_wr,ext_mem_req \
		-o reports/report_setup_$p-$sp ./rosko_setup 2000 2000 2000 $p $sp 1 1 1 pack $NTRIALS 1 orig;

		perf stat -e l2d_cache_refill,l2d_cache,bus_access_rd,bus_access_wr,ext_mem_req \
		-o reports/report_rosko_$p-$sp ./rosko_sgemm_test 2000 2000 2000 $p $sp 1 1 1 pack $NTRIALS 1 orig;

		perf stat -e l2d_cache_refill,l2d_cache,bus_access_rd,bus_access_wr,ext_mem_req \
		-o reports/report_armpl_$p-$sp ./arm_test 2000 2000 2000 $p $sp 1 1 1 pack $NTRIALS 1 orig;

	done
done


mv results results_arm_test; 
mv reports reports_arm_test;








# compile taco 
export TACO_PATH=$PWD/taco/build/lib;
export LD_LIBRARY_PATH=$TACO_PATH:$LD_LIBRARY_PATH

g++ -std=c++11 -fopenmp -pthread -O3 -DNDEBUG -DTACO -I taco/include -L$PWD/taco/build/lib taco_spmm.cpp -o taco_spmm -ltaco

echo "algo,M,K,N,p,sp,time" >> results


./arm_test 2048 2048 2048 $NCORES 0 $NTRIALS

for sp in 55 60 65 70 75 80 82 85 87 90 92 95 97 99 99.9
do
	./rosko_sgemm_test 2048 2048 2048 $NCORES $sp 1 1 1 pack $NTRIALS 0 orig
	./taco_spmm 2048 2048 2048 $NCORES 1 $sp
	rm rand.mtx
done


# for p in 2 3 4 5 6 7 8 9 10
# do
# 	for m in 10 96 111 960 2111
# 	do
# 		for k in 10 96 111 960 2111
# 		do
# 			for n in 10 96 111 960 2111
# 			do
				# ./write_sp_pack $m $k $n $p 50 1 1 1 pack orig
				# ./rosko_sgemm_test $m $k $n $p 50 1 1 1 pack 1 0 orig

# 			done
# 		done
# 	done
# done
