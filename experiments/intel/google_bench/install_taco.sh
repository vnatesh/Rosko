#!/bin/bash


git clone https://github.com/tensor-compiler/taco.git
cd taco
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DOPENMP=ON ..
make -j8

