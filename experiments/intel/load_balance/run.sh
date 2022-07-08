#!/bin/bash

#!/bin/bash

cd ../../../CAKE_on_CPU;
source env.sh;
cd ../experiments/intel/load_balance;
mkdir reports;

x=$PWD

export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9";



# compile rosko_test
make;

NTRIALS=1;
NCORES=10;

echo "algo,file,M,K,N,p,sparsity,time" >> load_balance
# run matmul bench through intel vtune 


for ((n=1; n <= $NCORES; n++));
do
	for file in data/*.mtx; 
	do
		./rosko_sgemm_reorder $file $n 1 $NTRIALS; 
	done

	for i in 80 87 95
		do
			./rosko_sgemm_test 10000 10000 10000 $n $i 1;
		done
	done

done

# python plots.py $NTRIALS;

# compile rosko
make;





# python plots.py $NTRIALS;
