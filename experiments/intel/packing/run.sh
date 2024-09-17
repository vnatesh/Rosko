#!/bin/bash
x=$PWD
cd ../../../CAKE_on_CPU;
source env.sh;
cd ..;
source env.sh;
cd $x;

echo $ROSKO_HOME;
echo $CAKE_HOME;

# compile mkl_sparse gemm test with Intel MKL
# cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
# make sointel64 examples="spblas/sparse_gemm" sycl_devices=cpu
# cp _results/intel64_so_tbb/spblas/sparse_gemm.out $x
# cd $x


# compile rosko
make;


echo "algo,store,M,K,N,sp,bw" >> result_pack

# for i in 80 82 85 87 90 92 95 97 99
for i in 80 87 95
do
	for n in {256..10240..512}
	do

		./rosko_sgemm_test $n $n 1 $i csr_file rosko_file 0;
		./rosko_sgemm_test $n $n 1 $i csr_file rosko_file 1;

	done
done

