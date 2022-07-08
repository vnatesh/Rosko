#!/bin/bash


# Download ML_Graph SuiteSparse matrices

pip install ssgetpy
mkdir data
cd data

python3 - <<END
import ssgetpy
ssgetpy.search(rowbounds=(5000,25000),colbounds=(5000,25000), \
    dtype = 'real', group='ML_Graph').download(destpath = '.', extract=True)
END

mv **/*.mtx .
# rm -R -- */;
rm *label*;