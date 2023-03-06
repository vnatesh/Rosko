#!/bin/bash


x=$PWD

# compile arm_test with ARMPL
# gcc -I/opt/arm/armpl_21.1_gcc-9.3/include -fopenmp  arm_test.c -o test.o  /opt/arm/armpl_21.1_gcc-9.3/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib -lm -o arm_test;

gcc -I/opt/arm/armpl_21.0_gcc-10.2/include -fopenmp  arm_test.c -o test.o \
  /opt/arm/armpl_21.0_gcc-10.2/lib/libarmpl_lp64_mp.a  -L{ARMPL_DIR}/lib \
  -lm -o arm_test;


make;

echo "layer,algo,M,K,N,sp,time" >> results

NCORES=4;
NTRIALS=10;

tail -n +2 layers.csv > mat_tmp.csv

INPUT=mat_tmp.csv
OLDIFS=$IFS
IFS=','
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
i=0


for sp in 70 80 90 95 98
do
	while read layer Cout Cin Wout Wf Win M K N
	do
		# echo $layer $Cout $Cin $Wout $Wf $Win $M $K $N 
		./rosko_sgemm_test $M $K $N $NCORES $sp $NTRIALS $layer
		./arm_test $M $K $N $NCORES $sp $NTRIALS $layer

		./svd_test $M $K $N $NCORES $sp $NTRIALS $layer

		# rm -rf mkl;
	done < $INPUT
	# IFS=$OLDIFS
done

rm mat_tmp.csv


