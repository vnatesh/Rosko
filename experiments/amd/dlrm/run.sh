#!/bin/bash


# compile inner product, rosko, and cake
make;

mkdir reports

export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";



NTRIALS=100
NCORES=10
# rm -rf rosko_prof;


for ((j=1; j <= $NCORES; j++));
do
	./rosko_sgemm_test 2024 246981 128 $j 99.9996 1 $NTRIALS;
	./rosko_sgemm_reorder 2024 246981 128 $j 99.9996 1 $NTRIALS;
done


# x=$PWD

# # compile mkl_sparse gemm test with Intel MKL
# cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
# make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu
# cp _results/intel64_so_tbb/spblas/sparse_gemm.out $x
# cd $x


# ./sparse_gemm.out 2024 246981 128 99.9996 1


# 	done
# done

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

