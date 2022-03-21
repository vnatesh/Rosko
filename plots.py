import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker

pip install ssgetpy

# Download SuiteSparse matrices
python - <<END
import ssgetpy
ssgetpy.search(rowbounds=(5000,22000),colbounds=(5000,22000), \
    dtype = 'real', group='ML_Graph').download(destpath = '.', extract=True)
ssgetpy.search(nzbounds=(35631,35633),\
    dtype = 'real', group='LPnetlib').download(destpath = '.', extract=True)
ssgetpy.search(nzbounds=(1853103,1853105),\
    dtype = 'real', group='Simon').download(destpath = '.', extract=True)
END


intel_color = '#0071c5'











def plot_rop_vs_arm_end_to_end(fname = 'rop_vs_arm'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['ROP', 'ARMPL','ARMCL']
	gflops_armpl=[0]*776;gflops_rop=[0]*776;dram_io_rop=[0]*776;dram_io_armpl=[0]*776;
	gflops_armcl=[0]*776; dram_io_armcl=[0]*776; dram_bw_rop=[0]*776;dram_bw_armpl=[0]*776;
	dram_bw_armcl=[0]*776;
	#
	df1 = pandas.read_csv('result_end_to_end')
	flops = []
	#
	# multiply by 64 bytes since external memory request non-cacheable 
	# and L2-data cache refills/writeback PMUs
	# in ARM are expressed in terms of number of cache lines
	nz = df1[(df1['algo'] == 'ROP_sp')]['nz']._values
	N = df1[(df1['algo'] == 'ROP_sp')]['N']._values
	flops = nz*N
	t_sp = df1[(df1['algo'] == 'ROP_sp')]['time']._values
	t_dense = df1[(df1['algo'] == 'ROP_dense')]['time']._values
	flops_sp = sum(flops) / sum(t_sp)
	flops_dense = sum(flops) / sum(t_dense)
		#
		#
		# if (dram_io_rop[i] < dram_io_armpl[i]) and (dram_io_rop[i] < dram_io_armcl[i]) \
		# and (gflops_rop[i] > gflops_armpl[i]) and (gflops_rop[i] > gflops_armcl[i]):
		# 	print(i)
		# #
	# plt.subplot(1, 2, 1)
	# flops = np.log10(np.array(flops))
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, gflops_rop, label = labels[0],  marker = markers[0], color = colors[5])
	plt.scatter(flops, gflops_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	plt.scatter(flops, gflops_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	#
	plt.title('(a) Throughput for spMM in Transformer Layers')
	plt.xlabel("number of ops (log10 scale)", fontsize = 18)
	plt.xticks(range(7,11))
	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	




def plot_rop_vs_arm_dnn(fname = 'rop_vs_arm'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['ROP', 'ARMPL','ARMCL']
	gflops_armpl=[0]*776;gflops_rop=[0]*776;dram_io_rop=[0]*776;dram_io_armpl=[0]*776;
	gflops_armcl=[0]*776; dram_io_armcl=[0]*776; dram_bw_rop=[0]*776;dram_bw_armpl=[0]*776;
	dram_bw_armcl=[0]*776;
	#
	df1 = pandas.read_csv('result_bench')
	flops = []
	#
	for i in range(776):
		# multiply by 64 bytes since external memory request non-cacheable 
		# and L2-data cache refills/writeback PMUs
		# in ARM are expressed in terms of number of cache lines
		nz = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['nz']._values[0]
		N = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['N']._values[0]
		M = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['M']._values[0]
		K = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['K']._values[0]
		#
		if (float(nz) / (M*K)) >= 0.8:
			continue
		#
		flops.append(nz*N)
		#
		a = open('reports_arm_trans/report_armpl_%d' % i,'r').read().split('\n')
		# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
		cpu_time = df1[(df1['algo'] == 'armpl') & (df1['id'] == i)]['time']._values[0]
		dram_io_armpl[i] = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		dram_io_armpl[i] += (((int(re.search(r'\d+', a[6]).group())*64.0))) / 1e9
		dram_bw_armpl[i] = dram_io_armpl[i] / cpu_time
		gflops_armpl[i] = (float(nz*N) / cpu_time) / (1e9)
		#
		a = open('reports_arm_trans/report_armcl_%d' % i,'r').read().split('\n')
		# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
		cpu_time = df1[(df1['algo'] == 'armcl') & (df1['id'] == i)]['time']._values[0]
		dram_io_armcl[i] = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		dram_io_armcl[i] += (((int(re.search(r'\d+', a[6]).group())*64.0))) / 1e9
		dram_bw_armcl[i] = dram_io_armcl[i] / cpu_time
		gflops_armcl[i] = (float(nz*N) / cpu_time) / (1e9)
		#
		a = open('reports_arm_trans/report_rop_%d' % i,'r').read().split('\n')
		# cpu_time1 = float(re.search(r'\d+\.\d+', a[8]).group())
		cpu_time = df1[(df1['algo'] == 'ROP') & (df1['id'] == i)]['time']._values[0]
		dram_io_rop[i] = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
		dram_io_rop[i] += (((int(re.search(r'\d+', a[6]).group())*64.0))) / 1e9
		dram_bw_rop[i] = dram_io_rop[i] / cpu_time
		gflops_rop[i] = (float(nz*N) / cpu_time) / (1e9)
		#
		# if (dram_io_rop[i] < dram_io_armpl[i]) and (dram_io_rop[i] < dram_io_armcl[i]) \
		# and (gflops_rop[i] > gflops_armpl[i]) and (gflops_rop[i] > gflops_armcl[i]):
		# 	print(i)
		# #
	# plt.subplot(1, 2, 1)
	flops = np.log10(np.array(flops))
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, gflops_rop, label = labels[0],  marker = markers[0], color = colors[5])
	plt.scatter(flops, gflops_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	plt.scatter(flops, gflops_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	#
	plt.title('(a) Throughput for spMM in Transformer Layers')
	plt.xlabel("number of ops (log10 scale)", fontsize = 18)
	plt.xticks(range(7,11))
	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, dram_io_rop, label = labels[0],  marker = markers[0], color = colors[5])
	plt.scatter(flops, dram_io_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	plt.scatter(flops, dram_io_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	#
	plt.title('(b) DRAM IO for spMM in Transformer Layers')
	plt.xlabel("number of ops (log10 scale)", fontsize = 18)
	plt.xticks(range(7,11))
	plt.ylabel("DRAM IO (GB)", fontsize = 18)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_io.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, dram_bw_rop, label = labels[0],  marker = markers[0], color = colors[5])
	plt.scatter(flops, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	plt.scatter(flops, dram_bw_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	#
	plt.title('(c) DRAM BW for spMM in Transformer Layers')
	plt.xlabel("number of ops (log10 scale)", fontsize = 18)
	plt.xticks(range(7,11))
	plt.ylabel("DRAM Bandwidth (GB/sec)", fontsize = 18)
	plt.legend(loc = "upper right", prop={'size': 12})
	plt.savefig("%s_bw.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	plt.figure(figsize = (6,4))
	plt.scatter(gflops_rop, dram_bw_rop, label = labels[0],  marker = markers[0], color = colors[5])
	plt.scatter(gflops_armpl, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[4])
	plt.scatter(gflops_armcl, dram_bw_armcl, label = labels[2],  marker = markers[3], color = colors[3])
	#
	plt.title('(d) BW Required to Attain Target Throughputs')
	plt.xlabel("Throughput (GFLOPs/sec)", fontsize = 18)
	# plt.xticks(range(7,11))
	plt.ylabel("DRAM Bandwidth (GB/sec)", fontsize = 18)
	plt.legend(loc = "upper right", prop={'size': 12})
	plt.savefig("%s_bw_tput.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_rop_vs_arm_dnn()





# def plot_cake_vs_arm_dnn_tput(fname = 'rop_vs_arm'):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','aqua','k','m','r']
# 	labels = ['ROP', 'ARMPL','ARMCL']
# 	#
# 	df1 = pandas.read_csv('result_bench')
# 	rop = ((df1[df1['algo'] == 'ROP']['N']*df1[df1['algo'] == 'ROP']['nz']) / df1[df1['algo'] == 'ROP']['time'])._values
# 	armpl = ((df1[df1['algo'] == 'armpl']['N']*df1[df1['algo'] == 'armpl']['nz']) / df1[df1['algo'] == 'armpl']['time'])._values
# 	armcl = ((df1[df1['algo'] == 'armcl']['N']*df1[df1['algo'] == 'armcl']['nz']) / df1[df1['algo'] == 'armcl']['time'])._values
# 	# flops = df1[df1['algo'] == 'ROP']['N']*df1[df1['algo'] == 'ROP']['nz']._values 	
# 	# rop=[]; armpl=[]; armcl=[];
# 	plt.figure(figsize = (6,4))
# 	N = 2048.0
# 	flops = np.log10(np.array(list(df1[df1['algo'] == 'ROP']['nz']*N)))
# 	plt.scatter(flops, rop / 1e9, label = labels[0],  marker = markers[0], color = colors[5])
# 	plt.scatter(flops, armpl / 1e9, label = labels[1],  marker = markers[1], color = colors[4])
# 	plt.scatter(flops, armcl / 1e9, label = labels[2],  marker = markers[3], color = colors[3])
# 	plt.title('Throughput for spMM in Transformer Layers')
# 	plt.xlabel("number of ops (log10 scale)", fontsize = 18)
# 	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
# 	# plt.xscale("log")
# 	# q = np.log10(np.array(list(set(df1[df1['algo'] == 'ROP']['nz']*N)))).astype(int)
# 	plt.xticks(range(7,11))
# 	plt.legend(labels = labels, loc = "upper left", prop={'size': 12})
# 	plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
# 	plt.show()
# 	plt.clf()
# 	plt.close('all')


# plot_cake_vs_arm_dnn_tput()


def plot_cake_vs_mkl_sparse(fname = 'cake_vs_mkl_sp'):
	plt.rcParams.update({'font.size': 12})
	# all matrices used are 99.87-99.97% sparse
	labels = ['Fash_mnist', \
	'har','indianpines','J_VowelsSmall', \
	'kmnist','mnist_test','optdigits',\
	'usps','worms20']
	df1 = pandas.read_csv('result_sp')
	rel_tput = df1[df1['algo'] == 'mkl']['time']._values / df1[df1['algo'] == 'cake']['time']._values
	rel_tput = rel_tput[1:]
	cake_bw = [12,11,10,10,10,12,12,7,7]
	mkl_bw = [8,17,15,13,8.5,12,6,32,24]
	X = np.arange(len(labels))
	# 
	plt.figure(figsize = (6,4))
	plt.title('(a) DRAM BW of SpMM in ROP vs MKL', fontsize = 18)
	plt.tick_params(labelbottom=False)   
	plt.bar(X + 0.00, cake_bw, color = 'r', width = 0.25)
	plt.bar(X + 0.25, mkl_bw, color = intel_color, width = 0.25)
	plt.ylabel("DRAM Bandwidth (GB/sec)", fontsize = 16)
	plt.legend(labels=['ROP', 'MKL'])
	plt.tight_layout()
	plt.savefig("%s_dram.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,5))
	plt.title('(b) Throughput of SpMM in ROP vs MKL', fontsize = 18)
	plt.bar(X + 0.00, rel_tput, color = 'r', width = 0.25)
	plt.bar(X + 0.25, 9*[1], color = intel_color, width = 0.25)
	plt.xticks(X, labels)
	plt.xticks(rotation=60)
	plt.ylabel("Throughput relative to MKL", fontsize = 16)
	plt.yticks(np.arange(0, 5, 1))
	plt.legend(labels=['ROP', 'MKL'])
	plt.tight_layout()
	plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')




plot_cake_vs_mkl_sparse()





def f(M,N,K,bw,comp):
	pack = 2.0*(M*K + K*N + M*N) / bw
	comp = float(M*N*K) / comp
	return pack / (pack + comp)


def plot_cake_shmoo():
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE', 'MKL','BLIS']
	#
	df1 = pandas.read_csv('shmoo2')
	cake = (df1[df1['algo'] == 'cake']['flops'] / df1[df1['algo'] == 'cake']['time'])._values
	mkl = (df1[df1['algo'] == 'mkl']['flops'] / df1[df1['algo'] == 'mkl']['time'])._values
	blis = (df1[df1['algo'] == 'blis']['flops'] / df1[df1['algo'] == 'blis']['time'])._values
	flops = df1[df1['algo'] == 'cake']['flops']._values / 1e6
	#
	plt.figure(figsize = (6,4))
	plt.scatter(flops, cake / 1e9, label = labels[0],  marker = markers[0], color = colors[0])
	# plt.scatter(flops, mkl / 1e9, label = labels[1],  marker = markers[0], color = colors[1])
	plt.scatter(flops, blis / 1e9, label = labels[2],  marker = markers[0], color = colors[2])
	#
	plt.title('Throughput For Small Matrices')
	plt.xlabel("number of ops (MFLOPs)", fontsize = 18)
	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
	# plt.xticks(f)
	plt.legend(loc = "lower right", prop={'size': 12})
	plt.savefig("shmoo2.pdf" , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_cake_shmoo()










def plot_cake_vs_arm_sparse(M,N,K,mc,kc,alpha,fname = 'cake_vs_arm_sp', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['ROP SpMM', 'ARMCL Dense MM']
	#
	sparsity = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
	cake_sp = [0.541215 / i for i in [0.710503, 0.633433, 0.555825, 0.475019, 0.390225, 0.303335, 0.208614, 0.116096]]
	arm_dense = [0.541215 / i for i in [0.541215, 0.541215, 0.541215, 0.541215, 0.541215, 0.541215, 0.541215, 0.541215]]
	plt.figure(figsize = (6,4))
	plt.plot(sparsity, cake_sp, label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(sparsity, arm_dense, label = labels[1],  marker = markers[0], color = colors[1])
	#
	plt.title('Speedup Over Dense MM For Different Sparsities')
	plt.xlabel("Sparsity", fontsize = 18)
	plt.ylabel("Speedup", fontsize = 18)
	plt.xticks(sparsity)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_cake_vs_arm_sparse(23040,23040,23040,144,144,1,ntrials=1)



def plot_cake_vs_mkl_sparse(M,N,K,mc,kc,alpha,fname = 'rop', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['ROP SpMM', 'MKL Dense MM','MKL SpMM']
	#
	sparsity = [0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]
	flops = [(((1-i)*10000*10000*10000) / 1e9) for i in sparsity]
	cake_sp = [2.096601, 1.704533, 1.280577, 0.827174, 0.417649, 0.363949, 0.291033, 0.269158, 0.219173]
	intel_sp = [7.265694, 6.943313, 6.024098, 5.401255, 1.139998, 0.583468, 0.133928, 0.078719, 0.041962]
	intel_dense = [1.689468 for i in sparsity]
	# cake_sp = [flops[i] / cake_sp[i] for i in range(len(sparsity))]
	# intel_sp = [flops[i] / intel_sp[i] for i in range(len(sparsity))]
	# intel_dense = [(10000*10000*10000 / 1e9) / intel_dense[i] for i in range(len(sparsity))]
	# cake_sp = [0.257055 / i for i in [0.278515, 0.228560, 0.173513, 0.113896, 0.061715]]
	# intel_dense = [0.257055 / i for i in [0.257055, 0.256688, 0.255590, 0.251398, 0.251183]]
	# intel_sp = [0.257055 / i for i in [0.937217, 0.853224, 0.816171, 0.676243, 0.221147]]
	plt.figure(figsize = (6,4))
	plt.plot(sparsity, cake_sp, label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(sparsity, intel_dense, label = labels[1],  marker = markers[0], color = colors[1])
	plt.plot(sparsity, intel_sp, label = labels[2],  marker = markers[0], color = intel_color)
	# plt.fill_between(sparsity[1:] + [0.9846], intel_dense[1:] + [(10000*10000*10000 / 1e9)/1.689468], cake_sp[1:] + [0.44], color=colors[1])
	#
	plt.title('Runtime vs Sparsity for Different GEMM Libraries')
	plt.xlabel("Sparsity", fontsize = 18)
	plt.ylabel("Runtime (sec)", fontsize = 18)
	# plt.xticks(sparsity)
	plt.legend(loc = "center left", prop={'size': 12})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_cake_vs_mkl_sparse(23040,23040,23040,144,144,1,ntrials=1)

