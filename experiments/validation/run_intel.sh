#!/bin/bash


make;
mkdir reports;

echo "layer,algo,M,K,N,p,sp,time" >> results

NCORES=10;
NTRIALS=10;

tail -n +2 layers.csv > mat_tmp.csv

INPUT=mat_tmp.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
i=0




# this seems to be the input generation for plots in Figure 8c
for sp in 80 85 90 95 98
do
	for p in 1 2 3 4 5 6 7 8 9 10
	do

		./rosko_io_pred 10000 10000 10000 $p $sp pack
		./write_sp_pack 10000 10000 10000 $p $sp 1 1 1 pack orig
		./rosko_setup 10000 10000 10000 $p $sp 1 1 1 pack $NTRIALS 0 orig
		./rosko_sgemm_test 10000 10000 10000 $p $sp 1 1 1 pack 1 0 orig


		vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_sgemm_result \
		$PWD/rosko_sgemm_test 10000 10000 10000 $p $sp 1 1 1 pack $NTRIALS 1 orig;

		vtune -report summary -r rosko_sgemm_result -format csv \
		-report-output reports/report_rosko_$p-$sp.csv -csv-delimiter comma;

		rm -rf rosko_sgemm_result;


		vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_setup_result \
		$PWD/rosko_setup 10000 10000 10000 $p $sp 1 1 1 pack $NTRIALS 1 orig;

		vtune -report summary -r rosko_setup_result -format csv \
		-report-output reports/report_setup_$p-$sp.csv -csv-delimiter comma;

		rm -rf rosko_setup_result;

	done
done


mv results results_intel_test; 
mv reports reports_intel_test;





# compile taco 
export TACO_PATH=$PWD/taco/build/lib;
export LD_LIBRARY_PATH=$TACO_PATH:$LD_LIBRARY_PATH

g++ -std=c++11 -fopenmp -pthread -O3 -DNDEBUG -DTACO -I taco/include -L$PWD/taco/build/lib taco_spmm.cpp -o taco_spmm -ltaco

# compile MKL-CSR
x=$PWD
sudo cp mkl_sparse_gemm.cpp /opt/intel/oneapi/mkl/2021.1.1/examples/sycl/spblas
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/mkl_sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/mkl_sparse_gemm.out $x
cd $x


# compile mkl dense
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c -Wl,--no-as-needed \
-L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread \
-lm -ldl -o mkl_sgemm_test

echo "algo,M,K,N,p,sp,time" >> results


# this seems to be the input generation for Figure 8a 
./mkl_sgemm_test 9984 9984 9984 $NCORES $NTRIALS;

for sp in 70 75 80 82 85 87 90 92 95 97 99 99.9
do
	./rosko_sgemm_test 9984 9984 9984 $NCORES $sp 1 1 1 pack $NTRIALS 0 orig
	./mkl_sparse_gemm.out 9984 9984 9984 $sp $NTRIALS;
	./taco_spmm 9984 9984 9984 $NCORES 1 $sp
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
