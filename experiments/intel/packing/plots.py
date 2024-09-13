import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker


def plot_rosko_vs_intel_pack(fname = 'rosko_vs_intel_pack'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','k','r','r']
	labels = ['CSR 80%', 'Rosko 80%', 'CSR 87%', 'Rosko 87%', 'CSR 95%', 'Rosko 95%']
	sparsity = [80, 87, 95]
	# N = range(768+1024,9985,512)
	# MAYA this range does not comply with the range on N in run.sh
	# N = range(256,9985,512)
	N = range(256, 10241, 512)
	dft = pandas.read_csv('result_pack')
	#
	#
	plt.figure(figsize = (6,4))
	for i in range(len(sparsity)):
		q = (dft[(dft['algo'] == 'mkl time') & (dft['sp'] == str(sparsity[i])) & (dft['store'] == 0)]['bw'].values \
		+ dft[(dft['algo'] == 'mkl time') & (dft['sp'] == sparsity[i]) & (dft['store'] == 1)]['bw'].values) / 2.0
		print("this is q: ", len(dft[(dft['algo'] == 'mkl time') & (dft['sp'] == str(sparsity[i])) & (dft['store'] == '0')]['bw'].values))
		print("this is N: ", len(N))
		plt.plot(N, q, label = labels[i*2], marker = markers[i], color = colors[0])
		q = (dft[(dft['algo'] == 'rosko time') & (dft['sp'] == sparsity[i]) & (dft['store'] == 0)]['bw'].values \
		+ dft[(dft['algo'] == 'rosko time') & (dft['sp'] == sparsity[i]) & (dft['store'] == 1)]['bw'].values) / 2.0
		plt.plot(N, q, label = labels[i*2+1],  marker = markers[i], color = colors[3])
	#
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(a) Packing Runtime in \nRosko vs MKL-CSR', fontsize = 24)
	plt.xlabel("N", fontsize = 24)
	plt.ylabel("Runtime (sec)", fontsize = 24)
	plt.xticks(range(0,10001,2000), fontsize = 18)
	plt.yticks( fontsize = 20)
	plt.legend(loc = "upper left", prop={'size': 14})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	for i in range(len(sparsity)):
		q = (dft[(dft['algo'] == 'mkl bw') & (dft['sp'] == sparsity[i]) & (dft['store'] == 0)]['bw'].values \
		+ dft[(dft['algo'] == 'mkl bw') & (dft['sp'] == sparsity[i]) & (dft['store'] == 1)]['bw'].values) / 2.0
		plt.plot(N, q, label = labels[i*2], marker = markers[i], color = colors[0])
		q = (dft[(dft['algo'] == 'rosko bw') & (dft['sp'] == sparsity[i]) & (dft['store'] == 0)]['bw'].values \
		+ dft[(dft['algo'] == 'rosko bw') & (dft['sp'] == sparsity[i]) & (dft['store'] == 1)]['bw'].values) / 2.0
		plt.plot(N, q, label = labels[i*2+1],  marker = markers[i], color = colors[3])
	#
	plt.title('(b) Packing DRAM Bandwidth Usage \n in Rosko vs MKL-CSR', fontsize = 24)
	plt.xlabel("N", fontsize = 24)
	plt.xticks(range(0,10001,2000), fontsize = 18)
	plt.yticks(np.arange(0,1.5,0.5), fontsize = 20)
	plt.ylabel("DRAM Bw (GB/s)", fontsize = 24)
	plt.legend(loc = "lower right", prop={'size': 14})
	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_rosko_vs_intel_pack()


