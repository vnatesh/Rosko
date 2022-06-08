#!/bin/bash


# Download ML_Graph SuiteSparse matrices

pip install ssgetpy

python3 - <<END
import ssgetpy
ssgetpy.search(rowbounds=(5000,45000),colbounds=(5000,45000), \
    dtype = 'real', group='ML_Graph').download(destpath = '.', extract=True)
END

mv **/*.mtx .
rm -R -- */;
rm *label*;


# export OMP_NUM_THREADS=24 
# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"         
NTRIALS=10


# compile cake_sp_sgemm_test
make

echo "algo,file,p,time" >> bar_load

for file in *.mtx; 
do

    ./aoclsparse-bench --function=csrmm --precision=d --alpha=1 --beta=0 --iters=$NTRIALS --mtx=$file > out;
value=$file python3 - <<END
import os
open('bar_load', 'a').write("aocl,%s,1,%s\n" % (os.environ["value"], \
float(open('out', 'r').read().split('\n')[2].split()[10]) / 1000.0)) 
END
    rm out;
    ./cake_spgemm_test $file $NTRIALS;

done

