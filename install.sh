#!/bin/bash

if [ -z ${CAKE_HOME+x} ]; 
then
	git clone https://github.com/vnatesh/CAKE_on_CPU.git
	cd CAKE_on_CPU
	source env.sh
	./install.sh
	make -f kernels.mk
	make
	sudo ldconfig $CAKE_HOME
	cd ..
else 
	echo "var is set to '$var'"; 
fi


if uname -m | grep -q 'aarch64'; 
then
   python3 $CAKE_HOME/python/kernel_gen.py armv8 20 72 sparse
   mv sparse.cpp src/kernels/armv8
else
	python3 $CAKE_HOME/python/kernel_gen.py haswell 8 32 sparse
   mv sparse.cpp src/kernels/haswell
fi


source ./env.sh
make -f rosko_kernels.mk
make

