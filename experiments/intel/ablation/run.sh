#!/bin/bash

mkdir reports;

# compile mkl_sgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 


# compile inner product, rosko, and cake
make;

for s in 80 82 85 87 90 92 95 97 99
do
	# MKL dense MM 
	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/prof_result \
	$PWD/mkl_sgemm_test 10000 10000 10000 10; 

	vtune -report summary -r prof_result -format csv \
	-report-output reports/report_mkl.csv -csv-delimiter comma;

	rm -rf prof_result;

	# CAKE dense MM
	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/cake_sgemm_result \
	$PWD/cake_sgemm_test 10000 10000 10000 10;

	vtune -report summary -r cake_sgemm_result -format csv \
	-report-output reports/report_cake_sgemm.csv -csv-delimiter comma;

	rm -rf cake_sgemm_result;


	# rosko with no ablation
	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/rosko_sgemm_result \
	$PWD/rosko_sgemm_test 10000 10000 10000 10 $s;

	vtune -report summary -r rosko_sgemm_result -format csv \
	-report-output reports/report_rosko_sgemm.csv -csv-delimiter comma;

	rm -rf rosko_sgemm_result;
done


# # inner product
# vtune --collect memory-access -data-limit=0 \
# -result-dir=$PWD/inner_prod_result \
# $PWD/inner_prod_test 5000 5000 5000 1;

# vtune -report summary -r inner_prod_result -format csv \
# -report-output reports/report_inner_prod.csv -csv-delimiter comma;

# # single-core rosko
# vtune --collect memory-access -data-limit=0 \
# -result-dir=$PWD/rosko_sgemm_result_single \
# $PWD/rosko_sgemm_test 5000 5000 5000 1;

# vtune -report summary -r rosko_sgemm_result_single -format csv \
# -report-output reports/report_rosko_sgemm_single.csv -csv-delimiter comma;


	
# rm -rf rosko_sgemm_result;


# python plots.py $NTRIALS;
