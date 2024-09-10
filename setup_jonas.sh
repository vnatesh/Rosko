# Remove CAKE if it's there
if [ -d "CAKE_on_CPU" ]; 
then
    sudo rm -rf CAKE_ON_CPU
else 
	echo "Hehe"; 
fi

# clean build as it is now for a fresh start
make clean


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