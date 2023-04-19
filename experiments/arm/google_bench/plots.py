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

def plot_rosko_vs_arm_dnn(fname = 'rosko_vs_arm_dlmc'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'm', 'k', 'b','g','aqua']
	labels = ['Rosko', 'ARMPL','ARMCL', 'TACO']
	gflops_armpl=[];gflops_rosko=[];dram_io_rosko=[];dram_io_armpl=[];
	gflops_armcl=[]; dram_io_armcl=[]; dram_bw_rosko=[];dram_bw_armpl=[];
	dram_bw_armcl=[];time_armcl=[]; time_armpl=[]; time_rosko=[];
	gflops_taco=[];dram_io_taco=[];time_taco=[];dram_bw_taco=[];
	sp = [70,80,90,95,98]
	armpl_st = dict(zip(sp,[[] for i in range(len(sp))]));
	armcl_st = dict(zip(sp,[[] for i in range(len(sp))]));
	taco_st = dict(zip(sp,[[] for i in range(len(sp))]))
	#
	df1 = pandas.read_csv('result_dlmc_arm')
	flops = []
	#
	for i in range(485):
		# multiply by 64 bytes since external memory request non-cacheable 
		# and L2-data cache refills/writeback PMUs
		# in ARM are expressed in terms of number of cache lines
		nz = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['nz']._values[0]
		N = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['N']._values[0]
		M = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['M']._values[0]
		K = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['K']._values[0]
		if M > 30000 or K > 30000 or N > 30000:
			continue
		# if (float(nz) / float(M*K)) >= 0.22:
		# 	continue
		# #
		# flops.append(100*(1.0 - (nz / float(M*K))))
		flops.append(nz)
		# if i in [96,193,290]:
		# 	# continue
		# 	print(M,N,K,nz,i)
		#
		a = open('reports_arm_trans/report_armpl_%d' % i,'r').read().split('\n')
		cpu_time = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['time']._values[0]
		tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		tmp -= 2*(M*K + K*N)*4 / 1e9 # read and write io due to A,B inits
		dram_io_armpl.append(tmp)
		dram_bw_armpl.append(tmp / cpu_time / 10)
		gflops_armpl.append((nz*N) / cpu_time / 1e9)
		time_armpl.append(cpu_time)
		#
		a = open('reports_arm_trans/report_armcl_%d' % i,'r').read().split('\n')
		cpu_time = df1[(df1['algo'] == 'armcl') & (df1['id'] == i)]['time']._values[0]
		tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		tmp -= 2*(M*K + K*N)*4 / 1e9
		dram_io_armcl.append(tmp)
		dram_bw_armcl.append(tmp / cpu_time / 10)
		gflops_armcl.append((nz*N) / (cpu_time) / 1e9)
		time_armcl.append(cpu_time)
		#
		a = open('reports_arm_trans/report_rosko_%d' % i,'r').read().split('\n')
		cpu_time = df1[(df1['algo'] == 'rosko') & (df1['id'] == i)]['time']._values[0]
		tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		tmp -= 2*(nz + K*N)*4 / 1e9
		dram_io_rosko.append(tmp)
		dram_bw_rosko.append(tmp / cpu_time / 10)
		gflops_rosko.append((nz*N) / cpu_time / 1e9)
		time_rosko.append(cpu_time)
		#
		a = open('reports_arm_trans/report_taco_%d' % i,'r').read().split('\n')
		try:
			cpu_time = df1[(df1['algo'] == 'taco') & (df1['id'] == i)]['time']._values[0]
		except IndexError:
			cpu_time = df1[(df1['algo'] == 'armcl') & (df1['id'] == i)]['time']._values[0]
		tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		tmp -= 2*(nz + K*N)*4 / 1e9
		a = open('reports_arm_trans/report_setup_taco_%d' % i,'r').read().split('\n')
		tmpq = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		tmp -= tmpq
		dram_io_taco.append(tmp)
		dram_bw_taco.append(tmp / cpu_time / 1)
		gflops_taco.append((nz*N) / cpu_time / 1e9)
		time_taco.append(cpu_time)
		if dram_bw_taco[-1] > 4:
			# dram_bw_taco[-1] = dram_bw_armcl[-1]
			dram_bw_taco[-1] = None
		# 
		#
		s = int(round_to_multiple(100.0 - 100.0*(float(nz) / float(M*K)), 1))
		armpl_st[s].append(gflops_rosko[-1]/gflops_armpl[-1])
		armcl_st[s].append(gflops_rosko[-1]/gflops_armcl[-1])
		taco_st[s].append(gflops_rosko[-1]/gflops_taco[-1])
		# i+=1
		# if (dram_io_rosko[i] < dram_io_mkl[i]) and (dram_io_rosko[i] < dram_io_armcl[i]) \
		# and (gflops_rosko[i] > gflops_armpl[i]) and (gflops_rosko[i] > gflops_armcl[i]):
		# 	print(i)
		# #
	flops = np.log10(np.array(flops))
	plt.figure(figsize = (6,4))
	plt.scatter(flops, gflops_rosko, label = labels[0],  marker = markers[0], color = colors[0], s=20)
	plt.scatter(flops, gflops_armpl, label = labels[1],  marker = markers[1], color = colors[1], s=20)
	plt.scatter(flops, gflops_armcl, label = labels[2],  marker = markers[3], color = colors[2], s=20)
	plt.scatter(flops, gflops_taco, label = labels[3],  marker = markers[2], color = colors[4], s=20)
	#
	plt.title('(b) Throughput for SpMM in\nTransformer Layers on ARM', fontsize = 24)
	plt.xlabel("# of nonzeroes (log scale)", fontsize = 24)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
	speedup = np.array(gflops_rosko) / np.array(gflops_armpl)
	print(gmean(speedup))
	print(stats.describe(speedup))
	print([i for i in speedup if i < 1.0 ])
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, dram_io_armpl, label = labels[1],  marker = markers[1], color = colors[1])
	plt.scatter(flops, dram_io_armcl, label = labels[2],  marker = markers[2], color = colors[2])
	plt.scatter(flops, dram_io_rosko, label = labels[0],  marker = markers[0], color = colors[0])
	plt.scatter(flops, dram_io_taco, label = labels[3],  marker = markers[3], color = colors[4])
	#
	plt.title('(a) DRAM IO for SpMM in\nTransformer Layers', fontsize = 24)
	plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.ylabel("DRAM IO (GB)", fontsize = 24)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_io.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[0])
	plt.scatter(flops, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[1])
	plt.scatter(flops, dram_bw_armcl, label = labels[2],  marker = markers[2], color = colors[2])
	plt.scatter(flops, dram_bw_taco, label = labels[3],  marker = markers[3], color = colors[4])
	#
	plt.title('(b) DRAM BW for SpMM in\nTransformer Layers', fontsize = 24)
	plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
	plt.legend(loc = "upper right", prop={'size': 12})
	plt.savefig("%s_bw.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	plt.figure(figsize = (6,4))
	plt.scatter(time_rosko, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[0])
	plt.scatter(time_armpl, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[1])
	plt.scatter(time_armcl, dram_bw_armcl, label = labels[2],  marker = markers[2], color = colors[2])
	plt.scatter(time_armcl, dram_bw_taco, label = labels[3],  marker = markers[3], color = colors[4])
	#
	plt.title('(c) BW Required to Attain\nTarget Runtime', fontsize = 24)
	plt.xlabel("Runtime (sec)", fontsize = 24)
	plt.xticks(fontsize = 16)
	plt.yticks(fontsize = 16)
	plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
	plt.legend(loc = "upper center", prop={'size': 12})
	plt.savefig("%s_bw_tput.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	armpl_st = {i: gmean(j) for i, j in armpl_st.items()}
	armcl_st = {i: gmean(j) for i, j in armcl_st.items()}
	taco_st = {i: gmean(j) for i, j in taco_st.items()}
	print(armpl_st)
	print(armcl_st)
	print(taco_st)
	plt.figure(figsize = (6,4))
	plt.plot(sp, [armpl_st[i] for i in sp], label = labels[1], marker = markers[1], color = colors[1])
	plt.plot(sp, [armcl_st[i] for i in sp], label = labels[2], marker = markers[2], color = colors[2])
	plt.plot(sp, [taco_st[i] for i in sp], label = labels[3], marker = markers[-1], color = colors[3])
	# plt.plot(sp, [1 for i in sp], color = 'k', linestyle = 'dashed')
	#
	plt.title('(d) GeoMean Speedup of Rosko\nOver Competitors on ARM', fontsize = 24)
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

	


plot_rosko_vs_arm_dnn()




