#!/bin/bash


# compile blis test
gcc -g -O2 -std=c99 -Wall -Wno-unused-function -Wfatal-errors -fPIC  \
-D_POSIX_C_SOURCE=200112L -fopenmp -I${CAKE_HOME}/include/blis \
-DBLIS_VERSION_STRING=\"0.8.1-67\" -I. -c blis_test.c -o blis_test.o;
g++ blis_test.o $CAKE_HOME/blis/lib/zen2/libblis-mt.a  -lm -lpthread -fopenmp \
-lrt -o blis_test;

# compile cake_sgemm_test
make;

export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23";

echo "algo,M,K,N,p,time" >> ml_workloads

p=24
ntrials=15


# remove csv header line
tail -n +2 matrix_sizes_ML_preliminary.csv > mat_tmp.csv

INPUT=mat_tmp.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read M K N
do

	./cake_sgemm_test $M $K $N $p $ntrials
	./blis_test $M $K $N $p $ntrials

done < $INPUT
IFS=$OLDIFS

rm mat_tmp.csv