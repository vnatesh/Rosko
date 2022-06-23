#!/bin/bash


# AOCL sparse installation

git clone https://github.com/amd/aocl-sparse.git
cd aocl-sparse
mkdir -p build/release
cd build/release
cmake ../..
make -j10
sudo make install

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$PWD/library
cd ../..
export AOCLSPARSE_ROOT=$PWD
cp library/include/aoclsparse_version.h.in library/include/aoclsparse_version.h
cd tests/benchmarks

# g++ aoclsparse_test.cpp ../common/*.cpp -I$AOCLSPARSE_ROOT/library/include \
# -I$AOCLSPARSE_ROOT/tests/include   \
# -L$LD_LIBRARY_PATH $AOCLSPARSE_ROOT/build/release/library/libaoclsparse.so \
# -o aoclsparse-bench

# compile aoclsparse-bench with CAKE
g++ -g -mavx -mfma -fopenmp aoclsparse_test.cpp ../common/*.cpp \
-I$AOCLSPARSE_ROOT/library/include \
-I$AOCLSPARSE_ROOT/tests/include   \
-I$CAKE_HOME/include  \
-L$LD_LIBRARY_PATH $AOCLSPARSE_ROOT/build/release/library/libaoclsparse.so \
-L$CAKE_HOME -lcake -o aoclsparse-bench


mv aoclsparse-bench ../../..
cd ../../..
