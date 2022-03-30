#!/bin/bash


x=$PWD

# compile mkl_sparse gemm test with Intel MKL
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/sparse_gemm.out $x
cd $x


# compile inner product, rosko, and cake
make;



for i in 80 82 85 87 90 92 95 97 99
do

	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/mkl_prof \
	$PWD/sparse_gemm.out 10000 10000 10000 $i 0;

	vtune -report summary -r mkl_prof -format csv \
	-report-output reports_pack/report_mkl.csv -csv-delimiter comma;

	rm -rf mkl_prof;

	# single-core rosko
	vtune --collect memory-access -data-limit=0 \
	-result-dir=$PWD/rosko_prof \
	$PWD/rosko_sgemm_test 10000 10000 10000 1 $i 0;

	vtune -report summary -r rosko_prof -format csv \
	-report-output reports_pack/report_rosko.csv -csv-delimiter comma;

	rm -rf rosko_prof;


	./sparse_gemm.out 10000 10000 10000 $i 1
	./rosko_sgemm_test 10000 10000 10000 1 $i 1;
done


# # inner product
# vtune --collect memory-access -data-limit=0 \
# -result-dir=$PWD/inner_prod_result \
# $PWD/inner_prod_test 5000 5000 5000 1;

# vtune -report summary -r inner_prod_result -format csv \
# -report-output reports_pack/report_inner_prod.csv -csv-delimiter comma;

# # single-core rosko
# vtune --collect memory-access -data-limit=0 \
# -result-dir=$PWD/rosko_sgemm_result_single \
# $PWD/rosko_sgemm_test 5000 5000 5000 1;

# vtune -report summary -r rosko_sgemm_result_single -format csv \
# -report-output reports_pack/report_rosko_sgemm_single.csv -csv-delimiter comma;


	
# rm -rf rosko_sgemm_result;


# python plots.py $NTRIALS;
