#!/bin/bash

mkdir reports_intel_ablate;


# # compile mkl_sgemm_test with Intel MKL
gcc -fopenmp -m64 -I${MKLROOT}/include mkl_sgemm_test.c \
-Wl,--no-as-needed -L${MKLROOT}/lib/intel64  \
-lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread \
 -lpthread -lm -ldl -o mkl_sgemm_test; 


# compile inner product, rosko, and cake
make;
NTRIALS=5
NCORES=10
export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";

echo "algo,M,K,N,sp,time" >> result_ablate_intel

# mkl dense MM
vtune --collect memory-access -data-limit=0 \
-result-dir=$PWD/mkl_sgemm_result \
$PWD/mkl_sgemm_test 10000 10000 10000 $NCORES 0 1;

vtune -report summary -r mkl_sgemm_result -format csv \
-report-output reports_intel_ablate/report_mkl.csv -csv-delimiter comma;

rm -rf mkl_sgemm_result;


# cake dense MM
vtune --collect memory-access -data-limit=0 \
-result-dir=$PWD/cake_sgemm_result \
$PWD/cake_sgemm_test 10000 10000 10000 $NCORES 0 1;

vtune -report summary -r cake_sgemm_result -format csv \
-report-output reports_intel_ablate/report_cake.csv -csv-delimiter comma;

rm -rf cake_sgemm_result;



vtune --collect uarch-exploration -data-limit=0 \
-result-dir=$PWD/mkl_sgemm_result \
$PWD/mkl_sgemm_test 10000 10000 10000 $NCORES 0 1;

vtune -report summary -r mkl_sgemm_result -format csv \
-report-output reports_intel_ablate/report_mkl_spec.csv -csv-delimiter comma;

rm -rf mkl_sgemm_result;


# cake dense MM
vtune --collect uarch-exploration -data-limit=0 \
-result-dir=$PWD/cake_sgemm_result \
$PWD/cake_sgemm_test 10000 10000 10000 $NCORES 0 1;

vtune -report summary -r cake_sgemm_result -format csv \
-report-output reports_intel_ablate/report_cake_spec.csv -csv-delimiter comma;

rm -rf cake_sgemm_result;



./mkl_sgemm_test 10000 10000 10000 $NCORES 1 $NTRIALS;
./cake_sgemm_test 10000 10000 10000 $NCORES 1 $NTRIALS;


for i in 80 82 85 87 90 92 95 97 99
do

	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/rosko_sgemm_result \
	$PWD/rosko_reorder 10000 10000 10000 $NCORES 0 $i 1;

	vtune -report summary -r rosko_sgemm_result -format csv \
	-report-output reports_intel_ablate/report_rosko_reorder_$i.csv -csv-delimiter comma;

	rm -rf rosko_sgemm_result;


	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/rosko_sgemm_result \
	$PWD/rosko_reorder_tile 10000 10000 10000 $NCORES 0 $i 1;

	vtune -report summary -r rosko_sgemm_result -format csv \
	-report-output reports_intel_ablate/report_rosko_reorder_tile_$i.csv -csv-delimiter comma;

	rm -rf rosko_sgemm_result;



	vtune --collect uarch-exploration -data-limit=0 \
	-result-dir=$PWD/rosko_sgemm_result \
	$PWD/rosko_reorder 10000 10000 10000 $NCORES 0 $i 1;

	vtune -report summary -r rosko_sgemm_result -format csv \
	-report-output reports_intel_ablate/report_rosko_reorder_spec_$i.csv -csv-delimiter comma;

	rm -rf rosko_sgemm_result;


	vtune --collect uarch-exploration -data-limit=0 \
	-result-dir=$PWD/rosko_sgemm_result \
	$PWD/rosko_reorder_tile 10000 10000 10000 $NCORES 0 $i 1;

	vtune -report summary -r rosko_sgemm_result -format csv \
	-report-output reports_intel_ablate/report_rosko_reorder_tile_spec_$i.csv -csv-delimiter comma;

	rm -rf rosko_sgemm_result;

	./rosko_reorder 10000 10000 10000 $NCORES 1 $i $NTRIALS
	./rosko_reorder_tile 10000 10000 10000 $NCORES 1 $i $NTRIALS

done




x=$PWD

mv $CAKE_HOME/src/kernels/haswell/sparse.cpp sparse_tmp.cpp
mv $CAKE_HOME/src/pack_ob.cpp pack_ob_tmp.cpp

mv sparse.cpp $CAKE_HOME/src/kernels/haswell
mv pack_ob.cpp $CAKE_HOME/src

cd $CAKE_HOME
make;
source env.sh;
cd $x;


for i in 80 82 85 87 90 92 95 97 99
do
	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/rosko_sgemm_result \
	$PWD/rosko 10000 10000 10000 $NCORES 0 $i 1;

	vtune -report summary -r rosko_sgemm_result -format csv \
	-report-output reports_intel_ablate/report_rosko_$i.csv -csv-delimiter comma;

	rm -rf rosko_sgemm_result;



	vtune --collect uarch-exploration -data-limit=0 \
	-result-dir=$PWD/rosko_sgemm_result \
	$PWD/rosko 10000 10000 10000 $NCORES 0 $i 1;

	vtune -report summary -r rosko_sgemm_result -format csv \
	-report-output reports_intel_ablate/report_rosko_spec_$i.csv -csv-delimiter comma;

	rm -rf rosko_sgemm_result;

	./rosko 10000 10000 10000 $NCORES 1 $i $NTRIALS

done


	
# rm -rf rosko_sgemm_result;


# python plots.py $NTRIALS;
