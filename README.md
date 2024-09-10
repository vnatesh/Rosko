# Overview
This repository contains performance experiments of row-skipping outer product kernels (Rosko) for sparse-dense matrix multiplication (spMM) on CPUs. Rosko uses the CAKE tiling and scheduling algorithm for efficient multicore parallelism and minimal DRAM bandwidth usage. Rosko kernels are written using SIMD intrinsics. They outperform Intel and ARM Dense GEMM and spMM libraries for matrices with sparsities ranging from 72% to 99.9%. 

## Jonas native installation:

the below stuff from Vikas is encapsulated in the following script, which also clean the directory from CAKE directory and C++ build before setting the whole thing up again:

```bash
./setup_jonas.sh
```

## Native Installation

```bash
# Download Rosko

# git clone https://github.com/vnatesh/Rosko.git (official repo, but not this fork)
git clone https://github.com/hidearmor/Rosko_jrya_mmhj.git
cd Rosko



# install CAKE

git clone https://github.com/vnatesh/CAKE_on_CPU.git
cd CAKE_on_CPU
source env.sh
./install.sh
make -f kernels.mk
make
sudo ldconfig $CAKE_HOME
cd ..




# Generate Rosko Kernels based on CPU arch. Only Haswell and armv8 currently supported

if uname -m | grep -q 'aarch64'; 
then
  	python3 $CAKE_HOME/python/kernel_gen.py armv8 20 72 sparse
  	mv sparse.cpp src/kernels/armv8
else
	python3 $CAKE_HOME/python/kernel_gen.py haswell 8 32 sparse
  	mv sparse.cpp src/kernels/haswell
fi



# install Rosko

source ./env.sh
make -f rosko_kernels.mk
make
sudo ldconfig $ROSKO_HOME
export LD_LIBRARY_PATH
```

Installation automatically downloads and installs the following tool/dependency verions:

* `CAKE` 


## Docker Installation

We provide a dockerfile to initialize a docker container with pre-installed Intel oneAPI and Rosko libraries.

```bash
sudo docker build . -t rosko_image
sudo docker run --pid=host --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE -it rosko_image
```


## Quick Start

In the `examples` directory, you will find a simple script `rosko_sgemm_test.cpp` that performs CAKE matrix multiplication on a random sparse MxK matrix and a dense KxN matrix, given M, K, and N values as command line arguments. To compile the script, simple type `make`and run the script as shown below. Make sure you have sourced the `env.sh` file before running. 

The script below runs 20 trials of an SpMM example on 8 cores for multiplying a 3000x2000 Sparse matrix with 90% sparsity and a 2000x1000 dense matrix (M = 3000, K = 2000, N = 1000)

```bash
(base) mmhj@DESKTOP-INVCLHF:~/ITU/Rosko_jrya_mmhj$ source ./env.sh
(base) mmhj@DESKTOP-INVCLHF:~/ITU/Rosko_jrya_mmhj$ cd examples/
(base) mmhj@DESKTOP-INVCLHF:~/ITU/Rosko_jrya_mmhj/examples$ make
g++ -I/home/mmhj/ITU/Rosko_jrya_mmhj/include -I/home/mmhj/ITU/Rosko_jrya_mmhj/CAKE_on_CPU/include -O3 -g -mavx -mfma -fopenmp rosko_sgemm_test.cpp -L/home/mmhj/ITU/Rosko_jrya_mmhj -lrosko -L/home/mmhj/ITU/Rosko_jrya_mmhj/CAKE_on_CPU -lcake -o rosko_sgemm_test
g++ -I/home/mmhj/ITU/Rosko_jrya_mmhj/include -I/home/mmhj/ITU/Rosko_jrya_mmhj/CAKE_on_CPU/include -O3 -g -mavx -mfma -fopenmp crisko_sgemm_test.cpp -L/home/mmhj/ITU/Rosko_jrya_mmhj -lrosko -L/home/mmhj/ITU/Rosko_jrya_mmhj/CAKE_on_CPU -lcake -o crisko_sgemm_test
(base) mmhj@DESKTOP-INVCLHF:~/ITU/Rosko_jrya_mmhj/examples$ ./rosko_sgemm_test 3000 2000 1000 8 90 20
M = 3000, K = 2000, N = 1000, cores = 8, sparsity = 0.900000
alg = 0, 4194304
sss 0.043422
sss 0.037009
sss 0.031292
sss 0.031281
sss 0.032200
sss 0.032261
sss 0.031886
sss 0.032135
sss 0.036221
sss 0.034001
sss 0.031604
sss 0.032400
sss 0.033888
sss 0.031955
sss 0.032900
sss 0.031075
sss 0.033897
sss 0.031196
sss 0.047687
sss 0.030360
sss 0.030745
sss 0.031935
sss 0.038252
sss 0.031491
sss 0.030727
sss 0.040589
sss 0.044013
sss 0.048948
sss 0.039204
sss 0.032765
0,90.000000,3000,2000,1000,0.035282
CORRECT!
```



## Running Experiments:

Before running experiments, make sure the following additional dependencies are installed:

* `Intel`
	* `Vtune 2021.1.1` 
	* `OpenMP 4.5` 
	* `Linux perf 5.4.86` 

* `ARM` 
	* `ARMPL 21.1` 
	* `ARMCL 22.02` 
	* `OpenMP 4.5` 
	* `Linux perf 5.4.86` 

The experiments are organized in separate directories for each CPU architecture tested (Intel and ARM). Each arch-specific directory contains sub-directories corresponding to figures in the Rosko paper. To run an experiment and plot the associated figure, simply enter the directory and execute the `run.sh` file. An example to generate the load balance figure for the Intel CPU tested is shown below. Experiments should be performed in `sudo` mode to enable permissions for hardware profiling.

```bash
~/Rosko$ sudo -s
~/Rosko$ source env.sh
~/Rosko$ cd experiments/intel/load_balance
~/Rosko/experiments/intel/load_balance$ ./run.sh
```

## Details
See paper for more details.
(<https://arxiv.org/pdf/2307.03930.pdf>).
<!-- <p align = "center">
<img  src="https://github.com/vnatesh/maestro/blob/master/images/cake_diagram.png" width="500">
</p>
 -->


