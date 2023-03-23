#!/bin/bash


# download and install TVM
git clone -b v0.7 --recursive https://github.com/apache/incubator-tvm tvm
cd tvm
git submodule init
git submodule update
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
mkdir build
cp cmake/config.cmake build
cd build

# modify build/config.cmake to add llvm directory (using tiramisu llvm)
sed -i.bak '/set(USE_LLVM OFF)/d' config.cmake
echo "set(USE_LLVM /home/vnatesh/tiramisu/3rdParty/llvm/build/bin/llvm-config)"  >> config.cmake

cmake ..
make -j4
cd ..


export TVM_HOME=$PWD
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export TVM_NUM_THREADS=10

cd python; python3.8 setup.py install --user; cd ..

pip3 install --user numpy decorator attrs
pip3 install --user tornado psutil 'xgboost<1.6.0' cloudpickle
pip3 install ogb


# download FeatGraph
cd ..
git clone https://github.com/amazon-science/FeatGraph.git
export PYTHONPATH=$PWD/FeatGraph/python:${PYTHONPATH}



# download and install deep graph library (dgl)
git clone --recurse-submodules https://github.com/dmlc/dgl.git
git submodule update --init --recursive
cd dgl
mkdir build
cd build
cmake ..
make -j4

cd ../python
python setup.py install
cd ../..
