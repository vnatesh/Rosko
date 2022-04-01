#!/bin/bash


# compile rosko
make;


# for i in 80 82 85 87 90 92 95 97 99
for i in 80 87 95
do
	for p in {1..10}
	do
		./rosko_sgemm_test 10000 10000 10000 $p $i 1;
	done
done


# python plots.py $NTRIALS;
