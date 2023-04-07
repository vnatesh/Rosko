import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker
from scipy import stats
from scipy.stats import gmean




def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)


def plot_rosko_vs_intel_dnn(fname = 'rosko_vs_intel_dlmc'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Rosko', 'MKL-Dense','MKL-CSR', 'Taco']
	gflops_mkl=[];gflops_rosko=[];
	gflops_mkl_sp=[]; gflops_taco=[];
	time_mkl_sparse=[]; time_mkl=[]; time_rosko=[]; time_taco=[]
	sp = [70,80,90,95,98]
	mkl_st = dict(zip(sp,[[] for i in range(len(sp))]));
	mkl_sp_st = dict(zip(sp,[[] for i in range(len(sp))]));
	taco_st = dict(zip(sp,[[] for i in range(len(sp))]))
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
		# if (float(nz) / float(M*K)) >= 0.22:
		# 	continue
		#
		# flops.append(100*(1.0 - (nz / float(M*K))))
		flops.append(nz)
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
		#
		s = int(round_to_multiple(100.0 - 100.0*(float(nz) / float(M*K)), 1))
		mkl_st[s].append(gflops_rosko[-1]/gflops_mkl[-1])
		mkl_sp_st[s].append(gflops_rosko[-1]/gflops_mkl_sp[-1])
		taco_st[s].append(gflops_rosko[-1]/gflops_taco[-1])
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
	plt.title('(a) Throughput for SpMM in\nTransformer Layers on Intel', fontsize = 24)
	plt.xlabel("# of nonzeroes (log scale)", fontsize = 24)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
	speedup = np.array(gflops_rosko) / np.array(gflops_mkl_sp)
	print("speedup over mkl_sparse = %f" %  gmean(speedup))
	print(stats.describe(speedup))
	print([i for i in speedup if i < 1.0 ])
	#
	speedup = np.array(gflops_rosko) / np.array(gflops_taco)
	print("speedup over taco = %f" %  gmean(speedup))
	print(stats.describe(speedup))
	#
	speedup = np.array(gflops_rosko) / np.array(gflops_mkl)
	print("speedup over mkl = %f" %  gmean(speedup))
	print(stats.describe(speedup))
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	mkl_st = {i: gmean(j) for i, j in mkl_st.items()}
	mkl_sp_st = {i: gmean(j) for i, j in mkl_sp_st.items()}
	taco_st = {i: gmean(j) for i, j in taco_st.items()}
	plt.figure(figsize = (6,4))
	plt.plot(sp, [mkl_st[i] for i in sp], label = labels[1], marker = markers[1], color = colors[4])
	plt.plot(sp, [mkl_sp_st[i] for i in sp], label = labels[2], marker = markers[3], color = colors[3])
	plt.plot(sp, [taco_st[i] for i in sp], label = labels[3], marker = markers[-1], color = colors[1])
	#
	plt.title('(c) GeoMean Speedup of Rosko\nover Competitors on Intel', fontsize = 24)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.xticks(fontsize = 16)
	plt.xlim(70,100)
	plt.yticks(fontsize = 16)
	plt.ylabel("GeoMean Speedup", fontsize = 18)
	plt.legend(loc = "center left", prop={'size': 12})
	plt.savefig("%s_mean.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#



plot_rosko_vs_intel_dnn()




