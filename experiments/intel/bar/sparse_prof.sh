#!/bin/bash

# cd ../../..;
# source env.sh;
# cd experiments/intel/Fig-10;
# mkdir reports;

x=$PWD


# Download ML_Graph SuiteSparse matrices
pip install ssgetpy

python - <<END
import ssgetpy
ssgetpy.search(rowbounds=(5000,22000),colbounds=(5000,22000), \
    dtype = 'real', group='ML_Graph').download(destpath = '.', extract=True)
ssgetpy.search(rowbounds=(5000,22000),colbounds=(5000,22000),\
dtype = 'real', group='Belcastro').download(destpath = '.', extract=True)
ssgetpy.search(nzbounds=(35631,35633),\
    dtype = 'real', group='LPnetlib').download(destpath = '.', extract=True)
ssgetpy.search(nzbounds=(1853103,1853105),\
    dtype = 'real', group='Simon').download(destpath = '.', extract=True)
END

mv **/*.mtx .
rm -R -- */;
rm *label*;


# compile mkl_sparse gemm test with Intel MKL
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/sparse_gemm.out $x
cd $x


# compile cake_sgemm_test
make;
mkdir reports;

NTRIALS=1;

# run matmul bench through intel vtune 
for file in *.mtx; 
do
	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/prof_result \
	 $PWD/sparse_gemm.out $file; 
	vtune -report summary -r prof_result -format csv \
		-report-output reports/report_mkl_$i-$j.csv -csv-delimiter comma;
	rm -rf prof_result;


	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/cake_sgemm_result \
		 $PWD/cake_spgemm_test $file;
	vtune -report summary -r cake_sgemm_result -format csv \
		-report-output reports/report_cake_sgemm_$i-$j.csv -csv-delimiter comma;
	rm -rf cake_sgemm_result;

	# ./sparse_gemm.out $file;
	# ./cake_spgemm_test $file; 
done


# python plots.py $NTRIALS;

