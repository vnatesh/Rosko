#!/bin/bash

git clone https://github.com/vnatesh/CAKE_on_CPU.git
cd CAKE_on_CPU
source ./env.sh
make
sudo ldconfig $CAKE_HOME
cd ..

