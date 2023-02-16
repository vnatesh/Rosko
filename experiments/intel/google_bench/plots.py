import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker



def plot_rosko_vs_intel_dnn(fname = 'rosko_vs_intel_dlmc'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Rosko', 'MKL-Dense','MKL-CSR', 'Taco']
	gflops_mkl=[];gflops_rosko=[];
	gflops_mkl_sp=[]; gflops_taco=[];
	time_mkl_sparse=[]; time_mkl=[]; time_rosko=[]; time_taco=[]
	#
	df1 = pandas.read_csv('result_dlmc')
	flops = []
	#
	for i in range(291,776):
		# multiply by 64 bytes since external memory request non-cacheable 
		# and L2-data cache refills/writeback PMUs
		# in ARM are expressed in terms of number of cache lines
		nz = df1[(df1['algo'] == 'mkl') & (df1['id'] == i)]['nz']._values[0]
		N = df1[(df1['algo'] == 'mkl') & (df1['id'] == i)]['N']._values[0]
		M = df1[(df1['algo'] == 'mkl') & (df1['id'] == i)]['M']._values[0]
		K = df1[(df1['algo'] == 'mkl') & (df1['id'] == i)]['K']._values[0]
		if M > 30000 or K > 30000 or N > 30000:
			continue
		#
		flops.append(nz*N)
		# if i in [96,193,290]:
		# 	# continue
		# 	print(M,N,K,nz,i)
		#
		cpu_time = df1[(df1['algo'] == 'mkl') & (df1['id'] == i)]['time']._values[0]
		gflops_mkl.append((nz*N) / cpu_time / 1e9)
		time_mkl.append(cpu_time)
		#
		cpu_time = df1[(df1['algo'] == 'mkl_sparse') & (df1['id'] == i)]['time']._values[0]
		# cpu_time /= 10
		gflops_mkl_sp.append((nz*N) / (cpu_time) / 1e9)
		time_mkl_sparse.append(cpu_time)
		#
		cpu_time = df1[(df1['algo'] == 'rosko') & (df1['id'] == i)]['time']._values[0]
		gflops_rosko.append((nz*N) / cpu_time / 1e9)
		time_rosko.append(cpu_time)
		#
		cpu_time = df1[(df1['algo'] == 'taco') & (df1['id'] == i)]['time']._values[0]
		gflops_taco.append((nz*N) / cpu_time / 1e9)
		time_taco.append(cpu_time)
		# i+=1
		# if (dram_io_rosko[i] < dram_io_mkl[i]) and (dram_io_rosko[i] < dram_io_mkl_sparse[i]) \
		# and (gflops_rosko[i] > gflops_mkl[i]) and (gflops_rosko[i] > gflops_mkl_sp[i]):
		# 	print(i)
		# #
	flops = np.log10(np.array(flops))
	plt.figure(figsize = (6,4))
	plt.scatter(flops, gflops_rosko, label = labels[0],  marker = markers[0], color = colors[5], s=20)
	plt.scatter(flops, gflops_mkl, label = labels[1],  marker = markers[1], color = colors[4], s=20)
	plt.scatter(flops, gflops_mkl_sp, label = labels[2],  marker = markers[3], color = colors[3], s=20)
	plt.scatter(flops, gflops_taco, label = labels[-1],  marker = markers[-1], color = colors[1], s=20)
	#
	plt.title('(a) Throughput for SpMM in\nTransformer Layers', fontsize = 24)
	plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
	# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#



plot_rosko_vs_intel_dnn()




