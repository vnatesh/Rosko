import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys




def plot_rosko_online(fname = 'rosko_online'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	# labels = ['CAKE', 'ONLINE', "ONLINE_BLIS", "single buf", "blis"]
	labels = ['rosko', 'online', 'online_B', 'online_BC', 'dense', 'ideal']
	# sp = 70
	for sp in [70,75,80,85,90,95,98,99]:
		df1 = pandas.read_csv('rosko_online')
		rosko = df1[(df1.algo == 0) & (df1.sp == sp)]['time'].values
		online = df1[(df1.algo == 1) & (df1.sp == sp)]['time'].values
		online_b = df1[(df1.algo == 2) & (df1.sp == sp)]['time'].values
		online_bc = df1[(df1.algo == 3) & (df1.sp == sp)]['time'].values
		dense = df1[(df1.algo == 4) & (df1.sp == 0)]['time'].values
		real = df1[(df1.algo == 5) & (df1.sp == sp)]['time'].values
		dims = df1[(df1.algo == 0) & (df1.sp == sp)]['M'].values
		#
		rosko_tput = [dims[i]**3 / rosko[i] / 1e9 for i in range(len(rosko))]
		onl_tput = [dims[i]**3 / online[i] / 1e9 for i in range(len(online))]
		onlb_tput = [dims[i]**3 / online_b[i] / 1e9 for i in range(len(online_b))]
		onlbc_tput = [dims[i]**3 / online_bc[i] / 1e9 for i in range(len(online_bc))]
		dense_tput = [dims[i]**3 / dense[i] / 1e9 for i in range(len(dense))]
		real_tput = [dims[i]**3 / real[i] / 1e9 for i in range(len(real))]
		plt.figure(figsize = (6,5))
		plt.plot(dims, rosko_tput, label = labels[0], color = colors[0])
		plt.plot(dims, onl_tput, label = labels[1], color = colors[1])
		plt.plot(dims, onlb_tput, label = labels[2], color = colors[2])
		plt.plot(dims, onlbc_tput, label = labels[3], color = colors[3])
		plt.plot(dims, dense_tput, label = labels[4], color = colors[4])
		plt.plot(dims, real_tput, label = labels[5], color = colors[5])
		#
		plt.title('Throughput For Rosko vs Online')
		plt.xlabel("M=N=K", fontsize = 24)
		plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 24)
		plt.legend(loc = "lower right", prop={'size': 16})
		plt.savefig("%s.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')	




plot_rosko_online()









def plot_rosko_online(fname = 'rosko_online'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	# labels = ['CAKE', 'ONLINE', "ONLINE_BLIS", "single buf", "blis"]
	labels = ['rosko', 'online_B', 'online_BC', 'online', 'dense', 'real']
	# sp = 70
	for sp in [70,75,80,85,90,95,98,99]:
		# df1 = pandas.read_csv('rosko_online')
		df1 = pandas.read_csv('rosko_online_arm')
		rosko = df1[(df1.algo == 0) & (df1.sp == sp)]['time'].values
		online = df1[(df1.algo == 1) & (df1.sp == sp)]['time'].values
		online_b = df1[(df1.algo == 2) & (df1.sp == sp)]['time'].values
		online_bc = df1[(df1.algo == 3) & (df1.sp == sp)]['time'].values
		dense = df1[(df1.algo == 4) & (df1.sp == 0)]['time'].values
		dims = df1[(df1.algo == 0) & (df1.sp == sp)]['M'].values
		#
		rosko_tput = [dims[i]**3 / rosko[i] / 1e9 for i in range(len(rosko))]
		onl_tput = [dims[i]**3 / online[i] / 1e9 for i in range(len(online))]
		onlb_tput = [dims[i]**3 / online_b[i] / 1e9 for i in range(len(online_b))]
		onlbc_tput = [dims[i]**3 / online_bc[i] / 1e9 for i in range(len(online_bc))]
		dense_tput = [dims[i]**3 / dense[i] / 1e9 for i in range(len(dense))]
		plt.figure(figsize = (6,5))
		plt.plot(dims, rosko_tput, label = labels[0], color = colors[0])
		plt.plot(dims, onl_tput, label = labels[1], color = colors[1])
		plt.plot(dims, onlb_tput, label = labels[2], color = colors[2])
		plt.plot(dims, onlbc_tput, label = labels[3], color = colors[3])
		plt.plot(dims, dense_tput, label = labels[4], color = colors[4])
		# plt.plot(dims, real_tput, label = labels[5], color = colors[5])
		#
		plt.title('Throughput For Rosko vs Online')
		plt.xlabel("M=N=K", fontsize = 24)
		plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 24)
		plt.legend(loc = "lower right", prop={'size': 16})
		plt.savefig("%s.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')	




plot_rosko_online()











def plot_rosko_online(fname = 'rosko_online_test'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	# labels = ['CAKE', 'ONLINE', "ONLINE_BLIS", "single buf", "blis"]
	labels = ['rosko', 'online test']
	# sp = 70
	for sp in [84]:
		# df1 = pandas.read_csv('rosko_online')
		df1 = pandas.read_csv('rosko_online_test')
		rosko = df1[(df1.algo == 0) & (df1.sp == sp)]['time'].values
		online = df1[(df1.algo == 1) & (df1.sp == sp)]['time'].values
		dims = df1[(df1.algo == 0) & (df1.sp == sp)]['M'].values
		#
		rosko_tput = [dims[i]**3 / rosko[i] / 1e9 for i in range(len(rosko))]
		onl_tput = [dims[i]**3 / online[i] / 1e9 for i in range(len(online))]
		plt.figure(figsize = (6,5))
		plt.plot(dims, rosko_tput, label = labels[0], color = colors[0])
		plt.plot(dims, onl_tput, label = labels[1], color = colors[1])
		#
		plt.title('Throughput For Rosko vs Online')
		plt.xlabel("M=N=K", fontsize = 24)
		plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 24)
		plt.legend(loc = "lower right", prop={'size': 16})
		plt.savefig("%s.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		speedup1 = [onl_tput[i] / rosko_tput[i] for i in range(len(rosko_tput))]	
		print("speedup over mkl = %f" %  gmean(speedup1))
		print(stats.describe(speedup1))




plot_rosko_online()



