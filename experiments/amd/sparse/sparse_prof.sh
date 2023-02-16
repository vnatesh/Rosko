#!/bin/bash


# export OMP_NUM_THREADS=24
# export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"        
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$PWD/aocl-sparse/build/release/library
NTRIALS=10

export GOMP_CPU_AFFINITY="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23";

# compile rosko_sgemm_test
make

echo "algo,file,p,time" >> bar_load
mkdir reports;

for file in data/*.mtx;
do

    ./aoclsparse-bench --function=csrmm --precision=d --alpha=1 --beta=0 --iters=$NTRIALS --mtx=${file##*/} > out;
value=${file##*/} python3 - <<END
import os
open('bar_load', 'a').write("aocl,%s,1,%s\n" % (os.environ["value"], \
float(open('out', 'r').read().split('\n')[2].split()[10]) / 1000.0))
END
    rm out;
    ./rosko_sgemm_test ${file##*/} $NTRIALS;


    # /home/vinatesh/AMDuProf_Linux_x64_3.5.671/bin/AMDuProfCLI collect \
    AMDuProfCLI collect \
    --event event=pmcx043,umask=0x08,interval=50000 \
    --event event=pmcx043,umask=0x40,interval=50000 \
    --event event=timer -o /tmp/test \
    ./rosko_sgemm_test ${file##*/} 1;

    # /home/vinatesh/AMDuProf_Linux_x64_3.5.671/bin/AMDuProfCLI report -i /tmp/test/*
    AMDuProfCLI report -i /tmp/test/*

    mv /tmp/test/*.csv reports/report_rosko_${file##*/}.csv;
    rm -rf /tmp/test;

    
    # /home/vinatesh/AMDuProf_Linux_x64_3.5.671/bin/AMDuProfCLI collect \
    AMDuProfCLI collect \
    --event event=pmcx043,umask=0x08,interval=50000 \
    --event event=pmcx043,umask=0x40,interval=50000 \
    --event event=timer -o /tmp/test \
    ./aoclsparse-bench --function=csrmm --precision=d --alpha=1 --beta=0 --iters=1 --mtx=${file##*/}

    # /home/vinatesh/AMDuProf_Linux_x64_3.5.671/bin/AMDuProfCLI report -i /tmp/test/*
    AMDuProfCLI report -i /tmp/test/*

    mv /tmp/test/*.csv reports/report_aocl_${file##*/}.csv;
    rm -rf /tmp/test;    

done
