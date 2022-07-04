#!/bin/bash

mkdir data;
cd data;

pip install ssgetpy

# python - <<END
# import ssgetpy
# import os
# import shutil

# ssgetpy.search(rowbounds=(2000,20000),colbounds=(2000,20000), \
#     dtype = 'real', limit = 100, group='ML_Graph').download(destpath = '.', extract=True)
# ssgetpy.search(rowbounds=(2000,20000),colbounds=(2000,20000), \
#     dtype = 'real', limit = 100, group='Newman').download(destpath = '.', extract=True)
# ssgetpy.search(rowbounds=(21100,21201),colbounds=(21100,21201), \
#     dtype = 'real', limit = 100, group='Simon').download(destpath = '.', extract=True)
# ssgetpy.search(rowbounds=(5000,15000),colbounds=(5000,15000), \
#     dtype = 'real', limit = 100, group='Mallya').download(destpath = '.', extract=True)


# ssgetpy.search(rowbounds=(13513,13515),colbounds=(13513,13515), \
#     dtype = 'real', limit = 100, group='FEMLAB').download(destpath = '.', extract=True)
# ssgetpy.search(rowbounds=(5000,16000),colbounds=(5000,16000), \
#     dtype = 'real', limit = 300).download(destpath = '.', extract=True)
# END


python - <<END
import ssgetpy
import os
import shutil

files = ssgetpy.search(rowbounds=(5000,16000),colbounds=(5000,16000), \
    dtype = 'real', limit = 300)
files.download(destpath = '.', extract=True)
for n in [files[i].name for i in range(len(files))]:
	try:
		shutil.move("%s/%s.mtx" % (n,n), ".")
		shutil.rmtree("%s" % n )
	except FileNotFoundError:
		continue
END




rm -R -- */;
