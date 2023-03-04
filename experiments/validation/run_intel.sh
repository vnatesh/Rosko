#!/bin/bash


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
