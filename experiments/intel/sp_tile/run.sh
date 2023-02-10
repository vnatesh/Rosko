#!/bin/bash


make;


export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";



# for x in {75..99..2};
# do
# 	./cake_sgemm_test 10000 10000 10000 10 $x 160 500 1440 test.txt 10
# done

echo "algo,sp,mr,nr,M,K,N,time" >> results_mr_nr


for x in 75 80 85 90 95 98 99 99.5;
do
	./cake_sgemm_test 10000 10000 10000 10 $x 10
done








