#!/bin/bash



export TVM_HOME=$PWD/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export PYTHONPATH=$PWD/FeatGraph/python:${PYTHONPATH}
export TVM_NUM_THREADS=10
# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";

x=$PWD
sudo cp mkl_sparse_gemm.cpp /opt/intel/oneapi/mkl/2021.1.1/examples/sycl/spblas
cd /opt/intel/oneapi/mkl/2021.1.1/examples/sycl
make sointel64 examples="spblas/mkl_sparse_gemm" sycl_devices=cpu
cp _results/intel64_so_tbb/spblas/mkl_sparse_gemm.out $x
cd $x


python3.8 write_csr.py
make
mkdir reports

# cp download_datasets.py FeatGraph/benchmark
# cd FeatGraph/benchmark
python3.8 download_datasets.py

NTRIALS=10;
NCORES=10


# cd ../..


echo "algo,file,M,K,N,p,sp,time" >> result_gnn

for len in 32 64 128 256 512;
do
	./write_rosko $len $NCORES reddit_data reddit_packed
	./write_rosko $len $NCORES ogbn-proteins ogbn_packed

	./rosko_sgemm_test $len $NCORES $NTRIALS reddit_packed 0 0
	./rosko_sgemm_test $len $NCORES $NTRIALS ogbn_packed 0 0

	python3.8 feat_test.py --dataset data/reddit_csr_float32.npz  \
	--feat-len $len --target x86 --nruns $NTRIALS --setup 0 > feat_reddit_$len

	python3.8 feat_test.py --dataset data/ogbn-proteins_csr.npz \
	--feat-len $len --target x86 --nruns $NTRIALS --setup 0 > feat_ogbn_$len 

	./mkl_sparse_gemm.out $len $NCORES $NTRIALS reddit_data 0 0
	./mkl_sparse_gemm.out $len $NCORES $NTRIALS ogbn-proteins 0 0





	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/rosko_sgemm_test $len $NCORES $NTRIALS reddit_packed 1 1;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_rosko_reddit_setup_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;


	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/rosko_sgemm_test $len $NCORES $NTRIALS ogbn_packed 1 1;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_rosko_ogbn_setup_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;



	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/rosko_sgemm_test $len $NCORES $NTRIALS reddit_packed 1 0;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_rosko_reddit_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;


	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/rosko_sgemm_test $len $NCORES $NTRIALS ogbn_packed 1 0;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_rosko_ogbn_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;






	vtune --collect memory-access -data-limit=0  \
	-result-dir=$PWD/rosko_result python3.8 feat_test.py \
	--dataset data/reddit_csr_float32.npz  --feat-len $len --target x86 --nruns $NTRIALS --setup 1
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_feat_reddit_setup_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;


	vtune --collect memory-access -data-limit=0  \
	-result-dir=$PWD/rosko_result python3.8 feat_test.py \
	--dataset data/ogbn-proteins_csr.npz --feat-len $len --target x86 --nruns $NTRIALS --setup 1 
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_feat_ogbn_setup_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;




	vtune --collect memory-access -data-limit=0  \
	-result-dir=$PWD/rosko_result python3.8 feat_test.py \
	--dataset data/reddit_csr_float32.npz  --feat-len $len --target x86 --nruns $NTRIALS --setup 0
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_feat_reddit_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;


	vtune --collect memory-access -data-limit=0  \
	-result-dir=$PWD/rosko_result python3.8 feat_test.py \
	--dataset data/ogbn-proteins_csr.npz --feat-len $len --target x86 --nruns $NTRIALS --setup 0
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_feat_ogbn_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;






	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/mkl_sparse_gemm.out $len $NCORES 1 reddit_packed 1 1;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_mkl_reddit_setup_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;


	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/mkl_sparse_gemm.out $len $NCORES 1 ogbn_packed 1 1;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_mkl_ogbn_setup_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;



	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/mkl_sparse_gemm.out $len $NCORES 1 reddit_packed 1 0;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_mkl_reddit_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;


	vtune --collect memory-access -data-limit=0 \
		-result-dir=$PWD/rosko_result \
		 $PWD/mkl_sparse_gemm.out $len $NCORES 1 ogbn_packed 1 0;
	vtune -report summary -r rosko_result -format csv \
		-report-output reports/report_mkl_ogbn_$len.csv -csv-delimiter comma;
	rm -rf rosko_result;
done


mv reports reports_gnn
