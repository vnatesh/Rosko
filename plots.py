import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys


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





def plot_cake_vs_arm_dnn():
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['ARMPL','ARMCL', 'ROP']
	#
	df1 = pandas.read_csv('result_bench')
	rop = ((df1[df1['algo'] == 'ROP']['N']*df1[df1['algo'] == 'ROP']['nz']) / df1[df1['algo'] == 'ROP']['time'])._values
	armpl = ((df1[df1['algo'] == 'armpl']['N']*df1[df1['algo'] == 'armpl']['nz']) / df1[df1['algo'] == 'armpl']['time'])._values
	armcl = ((df1[df1['algo'] == 'armcl']['N']*df1[df1['algo'] == 'armcl']['nz']) / df1[df1['algo'] == 'armcl']['time'])._values
	flops = df1[df1['algo'] == 'ROP']['N']*df1[df1['algo'] == 'ROP']['nz']._values 
	#
	# plt.scatter(flops, rop / 1e9, label = labels[0],  marker = markers[0], color = colors[0])
	# plt.scatter(flops, armpl / 1e9, label = labels[1],  marker = markers[0], color = colors[1])
	# plt.scatter(flops, armcl / 1e9, label = labels[2],  marker = markers[0], color = colors[2])
	#
	rop=[]; armpl=[]; armcl=[];
	N = 2048.0
	flops = np.log10(np.array(list(set(df1[df1['algo'] == 'ROP']['nz']*N))))
	#
	for i in set(df1[df1['algo'] == 'ROP']['nz']):
		rop.append((i*N / 1e9) / df1[(df1['algo']=='ROP') & (df1['nz']==i)]['time'].mean())
		armpl.append((i*N / 1e9) / df1[(df1['algo']=='armpl') & (df1['nz']==i)]['time'].mean())
		armcl.append((i*N / 1e9) / df1[(df1['algo']=='armcl') & (df1['nz']==i)]['time'].mean())
	#
	plt.figure(figsize = (6,4))
	#
	plt.bar(flops + 0, armpl, color = 'g', width = 0.05)
	plt.bar(flops + 0.04, armcl, color = intel_color, width = 0.05)
	plt.bar(flops + 0.08, rop, color = colors[-1], width = 0.05)
	plt.title('Throughput for spMM in Transformer Layers')
	plt.xlabel("number of ops (MFLOPs)", fontsize = 18)
	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
	plt.xscale("log")
	# plt.xticks(f)
	plt.legend(labels = labels, loc = "upper left", prop={'size': 12})
	plt.savefig("rop_vs_arm_dnn_bar.pdf" , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_cake_vs_arm_dnn()



def plot_cake_vs_arm_dnn():
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['ROP', 'ARMPL','ARMCL']
	#
	df1 = pandas.read_csv('result_bench')
	rop = ((df1[df1['algo'] == 'ROP']['N']*df1[df1['algo'] == 'ROP']['nz']) / df1[df1['algo'] == 'ROP']['time'])._values
	armpl = ((df1[df1['algo'] == 'armpl']['N']*df1[df1['algo'] == 'armpl']['nz']) / df1[df1['algo'] == 'armpl']['time'])._values
	armcl = ((df1[df1['algo'] == 'armcl']['N']*df1[df1['algo'] == 'armcl']['nz']) / df1[df1['algo'] == 'armcl']['time'])._values
	flops = df1[df1['algo'] == 'ROP']['N']*df1[df1['algo'] == 'ROP']['nz']._values 
	#
	# plt.scatter(flops, rop / 1e9, label = labels[0],  marker = markers[0], color = colors[0])
	# plt.scatter(flops, armpl / 1e9, label = labels[1],  marker = markers[0], color = colors[1])
	# plt.scatter(flops, armcl / 1e9, label = labels[2],  marker = markers[0], color = colors[2])
	#
	plt.figure(figsize = (6,4))
	plt.stem(flops, rop / 1e9, markerfmt=' ', basefmt=" ", linefmt = 'r')
	plt.title('ROP')
	plt.xlabel("number of ops (MFLOPs)", fontsize = 18)
	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
	plt.xscale("log")
	plt.yticks(range(6))
	plt.savefig("rop_dnn.pdf" , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.stem(flops, armpl / 1e9, markerfmt=' ', basefmt=" ", linefmt = colors[1])
	plt.title('ARMPL')
	plt.xlabel("number of ops (MFLOPs)", fontsize = 18)
	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
	plt.xscale("log")
	plt.yticks(range(6))
	plt.savefig("armpl_dnn.pdf" , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.stem(flops, armcl / 1e9, markerfmt=' ', basefmt=" ", linefmt = colors[0])
	plt.title('ARMCL')
	plt.xlabel("number of ops (MFLOPs)", fontsize = 18)
	plt.ylabel("Throughput (GLOPs/sec)", fontsize = 18)
	plt.xscale("log")
	plt.yticks(range(6))
	plt.savefig("armcl_dnn.pdf" , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')

plot_cake_vs_arm_dnn()








def plot_cake_vs_mkl_sparse(fname = 'cake_vs_mkl_sp'):
	plt.rcParams.update({'font.size': 12})
	# all matrices used are 99.87-99.97% sparse
	labels = ['Fashion_mnist', \
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
	plt.title('DRAM Bandwidth of SpMM in CAKE vs MKL', fontsize = 18)
	plt.tick_params(labelbottom=False)   
	plt.bar(X + 0.00, cake_bw, color = 'r', width = 0.25)
	plt.bar(X + 0.25, mkl_bw, color = intel_color, width = 0.25)
	plt.ylabel("DRAM Bandwidth (GB/sec)", fontsize = 16)
	plt.legend(labels=['CAKE', 'MKL'])
	plt.tight_layout()
	plt.savefig("%s_dram.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,5))
	plt.bar(X + 0.00, rel_tput, color = 'r', width = 0.25)
	plt.bar(X + 0.25, 9*[1], color = intel_color, width = 0.25)
	plt.xticks(X, labels)
	plt.xticks(rotation=60)
	plt.ylabel("Throughput relative to MKL", fontsize = 16)
	plt.yticks(np.arange(0, 5, 1))
	plt.legend(labels=['CAKE', 'MKL'])
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



def plot_cake_vs_mkl_sparse(M,N,K,mc,kc,alpha,fname = 'cake_vs_mkl_sp', ntrials=10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['CAKE SpMM', 'MKL Dense MM','MKL SpMM']
	#
	sparsity = [0.8, 0.85, 0.9, 0.95, 0.99]
	cake_sp = [0.257055 / i for i in [0.278515, 0.228560, 0.173513, 0.113896, 0.061715]]
	intel_dense = [0.257055 / i for i in [0.257055, 0.256688, 0.255590, 0.251398, 0.251183]]
	intel_sp = [0.257055 / i for i in [0.937217, 0.853224, 0.816171, 0.676243, 0.221147]]
	plt.figure(figsize = (6,4))
	plt.plot(sparsity, cake_sp, label = labels[0],  marker = markers[0], color = colors[0])
	plt.plot(sparsity, intel_dense, label = labels[1],  marker = markers[0], color = colors[1])
	plt.plot(sparsity, intel_sp, label = labels[2],  marker = markers[0], color = intel_color)
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


plot_cake_vs_mkl_sparse(23040,23040,23040,144,144,1,ntrials=1)

