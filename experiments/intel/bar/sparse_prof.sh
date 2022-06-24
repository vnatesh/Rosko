#!/bin/bash

cd ../../../CAKE_on_CPU;
source env.sh;
cd ../experiments/intel/bar;
mkdir reports;

x=$PWD


# Download ML_Graph SuiteSparse matrices
pip install ssgetpy



python - <<END
import ssgetpy
ssgetpy.search(rowbounds=(5000,22000),colbounds=(5000,22000), \
    dtype = 'real', group='ML_Graph').download(destpath = '.', extract=True)
END

# python - <<END
# import ssgetpy
# ssgetpy.search(rowbounds=(5000,22000),colbounds=(5000,22000), \
#     dtype = 'real', group='ML_Graph').download(destpath = '.', extract=True)
# # ssgetpy.search(rowbounds=(5000,22000),colbounds=(5000,22000),\
# # dtype = 'real', group='Belcastro').download(destpath = '.', extract=True)
# ssgetpy.search(nzbounds=(35631,35633),\
#     dtype = 'real', group='LPnetlib').download(destpath = '.', extract=True)
# ssgetpy.search(nzbounds=(1853103,1853105),\
#     dtype = 'real', group='Simon').download(destpath = '.', extract=True)
# END

mv **/*.mtx .
rm -R -- */;
rm *label*;


# compile mkl_sparse gemm test with Intel MKL
sudo cp sparse_gemm.cpp /opt/intel/oneapi/mkl/2021.1.1/examples/sycl/spblas
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/sparse_gemm.out $x
cd $x


# compile cake_sgemm_test
make;

NTRIALS=10;
NCORES=10;

echo "algo,file,p,time" >> result_sp
# run matmul bench through intel vtune 


for ((n=10; n <= $NCORES; n++));
do
	for file in *.mtx; 
	do
		vtune --collect memory-access -data-limit=0 \
			-result-dir=$PWD/prof_result \
		 	$PWD/sparse_gemm.out $file $n 0; 
		vtune -report summary -r prof_result -format csv \
			-report-output reports/report_mkl_$i-$j.csv -csv-delimiter comma;
		rm -rf prof_result;


		vtune --collect memory-access -data-limit=0 \
			-result-dir=$PWD/cake_sgemm_result \
			 $PWD/cake_spgemm_test $file $n 0;
		vtune -report summary -r cake_sgemm_result -format csv \
			-report-output reports/report_cake_sgemm_$i-$j.csv -csv-delimiter comma;
		rm -rf cake_sgemm_result;

		./sparse_gemm.out $file $n 1 $NTRIALS;
		./cake_spgemm_test $file $n 1 $NTRIALS; 
	done
done

# python plots.py $NTRIALS;

