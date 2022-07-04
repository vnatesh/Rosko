import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker



def plot_rosko_vs_arm_dnn(fname = 'rosko_vs_arm'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Rosko', 'ARMPL','ARMCL', 'CAKE']
	gflops_armpl=[0]*387;gflops_rosko=[0]*387;dram_io_rosko=[0]*387;dram_io_armpl=[0]*387;
	gflops_armcl=[0]*387; dram_io_armcl=[0]*387; dram_bw_rosko=[0]*387;dram_bw_armpl=[0]*387;
	dram_bw_armcl=[0]*387;dram_io_cake=[0]*387; dram_bw_cake=[0]*387; gflops_cake=[0]*387
	#
	df1 = pandas.read_csv('result_dlmc')
	flops = []
	#
	for i in range(387):
		# multiply by 64 bytes since external memory request non-cacheable 
		# and L2-data cache refills/writeback PMUs
		# in ARM are expressed in terms of number of cache lines
		nz = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['nz']._values[0]
		N = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['N']._values[0]
		M = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['M']._values[0]
		K = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['K']._values[0]
		#
		# sparsity = (1 - (float(nz) / (M*K)))
		# if sparsity <= 0.85 and sparsity >= 0.76:
		# 	continue
		#
		if M > 30000 or K > 30000 or N > 30000:
			flops.append(0)
			continue
		# if i in [96,193,290]:
			# flops.append(0)
			# continue
		#
		flops.append(nz)
		# if i in [96,193,290]:
		# 	# continue
		# 	print(M,N,K,nz,i)
		#
		# a = open('reports_arm_trans/report_armpl_%d' % i,'r').read().split('\n')
		cpu_time = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['time']._values[0]
		# dram_io_armpl[i] = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		# dram_io_armpl[i] += (((int(re.search(r'\d+', a[6]).group())*64.0))) / 1e9
		# dram_io_armpl[i] -= 2*(M*K + K*N)*4 / 1e9 # read and write io due to A,B inits
		# dram_bw_armpl[i] = dram_io_armpl[i] / cpu_time
		gflops_armpl[i] = (nz*N) / cpu_time
		#
		# a = open('reports_arm_trans/report_armcl_%d' % i,'r').read().split('\n')
		cpu_time = df1[(df1['algo'] == 'armcl') & (df1['id'] == i)]['time']._values[0]
		# dram_io_armcl[i] = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		# dram_io_armcl[i] += (((int(re.search(r'\d+', a[6]).group())*64.0))) / 1e9
		# dram_io_armcl[i] -= 2*(M*K + K*N)*4 / 1e9
		# dram_bw_armcl[i] = dram_io_armcl[i] / cpu_time
		gflops_armcl[i] = (nz*N) / cpu_time
		#
		# a = open('reports_arm_trans/report_rop_%d' % i,'r').read().split('\n')
		cpu_time = df1[(df1['algo'] == 'rosko') & (df1['id'] == i)]['time']._values[0]
		# dram_io_rosko[i] = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		# dram_io_rosko[i] += (((int(re.search(r'\d+', a[6]).group())*64.0))) / 1e9
		# dram_io_rosko[i] -= 2*(nz + K*N)*4 / 1e9
		# dram_bw_rosko[i] = dram_io_rosko[i] / cpu_time
		gflops_rosko[i] = (nz*N) / cpu_time
		#
		# a = open('reports_arm_trans/report_cake_%d' % i,'r').read().split('\n')
		# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
		cpu_time = df1[(df1['algo'] == 'CAKE') & (df1['id'] == i)]['time']._values[0]
		# dram_io_cake[i] = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		# dram_io_cake[i] += (((int(re.search(r'\d+', a[6]).group())*64.0))) / 1e9
		# dram_io_cake[i] -= 2*(nz + K*N)*4 / 1e9
		# dram_bw_cake[i] = dram_io_cake[i] / cpu_time
		gflops_cake[i] = (nz*N) / cpu_time
		# if (dram_io_rosko[i] < dram_io_armpl[i]) and (dram_io_rosko[i] < dram_io_armcl[i]) \
		# and (gflops_rosko[i] > gflops_armpl[i]) and (gflops_rosko[i] > gflops_armcl[i]):
		# 	print(i)
		# #
	# plt.subplot(1, 2, 1)
	flops = np.log10(np.array(flops))
	flops = [i for i in flops if i !=0 and i != np.inf and i != -np.inf]	
	gflops_armcl = [i for i in gflops_armcl if i !=0 and i != np.inf and i != -np.inf]
	gflops_armpl = [i for i in gflops_armpl if i !=0 and i != np.inf and i != -np.inf]
	gflops_cake = [i for i in gflops_cake if i !=0 and i != np.inf and i != -np.inf]
	gflops_rosko = [i for i in gflops_rosko if i !=0 and i != np.inf and i != -np.inf]
	# dram_io_armcl = [i for i in dram_io_armcl if i !=0 and i != np.inf and i != -np.inf]
	# dram_io_armpl = [i for i in dram_io_armpl if i !=0 and i != np.inf and i != -np.inf]
	# dram_io_cake = [i for i in dram_io_cake if i !=0 and i != np.inf and i != -np.inf]
	# dram_io_rosko = [i for i in dram_io_rosko if i !=0 and i != np.inf and i != -np.inf]
	# dram_bw_armcl = [i for i in dram_bw_armcl if i !=0 and i != np.inf and i != -np.inf]
	# dram_bw_armpl = [i for i in dram_bw_armpl if i !=0 and i != np.inf and i != -np.inf]
	# dram_bw_cake = [i for i in dram_bw_cake if i !=0 and i != np.inf and i != -np.inf]
	# dram_bw_rosko = [i for i in dram_bw_rosko if i !=0 and i != np.inf and i != -np.inf]
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, gflops_rosko, label = labels[0],  marker = markers[0], color = colors[5])
	plt.scatter(flops, gflops_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	plt.scatter(flops, gflops_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	plt.scatter(flops, gflops_cake, label = labels[-1],  marker = markers[-1], color = colors[1])
	#
	plt.title('(a) Throughput for SpMM in\nTransformer Layers', fontsize = 18)
	plt.xlabel("# of nonzeros (log10 scale)", fontsize = 18)
	plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	# plt.figure(figsize = (6,4))
	# plt.scatter(flops, dram_io_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	# plt.scatter(flops, dram_io_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	# plt.scatter(flops, dram_io_cake, label = labels[-1],  marker = markers[-1], color = colors[1])
	# plt.scatter(flops, dram_io_rosko, label = labels[0],  marker = markers[0], color = colors[5])
	# #
	# plt.title('(b) DRAM IO for SpMM in\nTransformer Layers', fontsize = 24)
	# plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
	# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
	# plt.yticks(fontsize = 16)
	# plt.ylabel("DRAM IO (GB)", fontsize = 24)
	# plt.legend(loc = "upper left", prop={'size': 12})
	# plt.savefig("%s_io.pdf" % fname, bbox_inches='tight')
	# plt.show()
	# plt.clf()
	# plt.close('all')
	# #
	# #
	# plt.figure(figsize = (6,4))
	# plt.scatter(flops, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[5])
	# plt.scatter(flops, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	# plt.scatter(flops, dram_bw_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	# plt.scatter(flops, dram_bw_cake, label = labels[-1],  marker = markers[-1], color = colors[1])
	# #
	# plt.title('(c) DRAM BW for SpMM in\nTransformer Layers', fontsize = 24)
	# plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
	# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
	# plt.yticks(fontsize = 16)
	# plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
	# plt.legend(loc = "upper right", prop={'size': 12})
	# plt.savefig("%s_bw.pdf" % fname, bbox_inches='tight')
	# plt.show()
	# plt.clf()
	# plt.close('all')
	# #
	# #
	# plt.figure(figsize = (6,4))
	# plt.scatter(gflops_armpl, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	# plt.scatter(gflops_armcl, dram_bw_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	# plt.scatter(gflops_cake, dram_bw_cake, label = labels[-1],  marker = markers[-1], color = colors[1])
	# plt.scatter(gflops_rosko, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[5])
	# #
	# plt.title('(d) BW Required to Attain\nTarget Runtime', fontsize = 24)
	# plt.xlabel("Runtime (sec)", fontsize = 24)
	# plt.xticks(np.arange(0,0.31,0.05), fontsize = 16)
	# plt.yticks(fontsize = 16)
	# plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
	# plt.legend(loc = "upper right", prop={'size': 12})
	# plt.savefig("%s_bw_tput.pdf" % fname, bbox_inches='tight')
	# plt.show()
	# plt.clf()
	# plt.close('all')


plot_rosko_vs_arm_dnn()




