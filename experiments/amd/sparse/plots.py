import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker





def plot_rosko_vs_aocl_sparse(fname = 'rosko_vs_aocl_sp'):
	plt.rcParams.update({'font.size': 12})
	# all matrices used are 99.87-99.97% sparse
	labels = ['Fash_mnist', \
	'har','indianpines','J_VowelsSmall', \
	'kmnist','mnist_test','optdigits',\
	'usps','worms20']
	df1 = pandas.read_csv('bar_load')
	# labels = [i[:-4] for i in df1[df1['algo'] == 'aocl']['file']._values]
	rel_tput = df1[df1['algo'] == 'aocl']['time']._values / df1[df1['algo'] == 'rosko']['time']._values
	X = np.arange(len(labels))
	#
	plt.figure(figsize = (6,5))
	plt.title('(a) Throughput of SpMM in Rosko vs aocl', fontsize = 18)
	plt.bar(X + 0.00, rel_tput, color = 'r', width = 0.25)
	plt.bar(X + 0.25, len(labels)*[1], color = 'g', width = 0.25)
	plt.xticks(X, labels, fontsize = 18)
	plt.xticks(rotation=60)
	plt.ylabel("Tput relative to aocl", fontsize = 16)
	# plt.yticks(np.arange(0, 5, 1), fontsize = 16)
	plt.legend(labels=['Rosko', 'AOCL'])
	plt.tight_layout()
	plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	#
	dram_bw_rosko = []
	dram_bw_aocl = []
	#
	for i in df1[df1['algo'] == 'aocl']['file']._values:
		a = open("reports_amd/report_rosko_%s.csv" % i ,'r').read()
		data = [q for q in a.split('\n') if 'PID' in q][0].split(',')
		cpu_time = df1[(df1['algo'] == 'rosko') & (df1['file'] == i)]['time']._values.mean()
		dram_bw_rosko.append((((float(data[2]) + float(data[3]))*50000*64) / cpu_time) / 1e9)
		#
		a = open("reports_amd/report_aocl_%s.csv" % i ,'r').read()
		data = [q for q in a.split('\n') if 'PID' in q][0].split(',')
		cpu_time = df1[(df1['algo'] == 'aocl') & (df1['file'] == i)]['time']._values.mean()
		dram_bw_aocl.append((((float(data[2]) + float(data[3]))*50000*64) / cpu_time) / 1e9)
	#
	#
	X = np.arange(len(labels))
	plt.figure(figsize = (6,5))
	plt.title('(b) DRAM BW of SpMM in Rosko vs AOCL', fontsize = 18)
	# plt.tick_params(labelbottom=False)  
	plt.bar(X + 0.00, dram_bw_rosko, color = 'r', width = 0.25)
	plt.bar(X + 0.25, dram_bw_aocl, color = 'g', width = 0.25)
	plt.ylabel("DRAM Bw (GB/sec)", fontsize = 16)
	plt.xticks(X, labels, fontsize = 18)
	plt.xticks(rotation=60)
	plt.yticks( fontsize = 16)
	plt.legend(labels=['Rosko', 'AOCL'])
	plt.tight_layout()
	plt.savefig("%s_dram.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_rosko_vs_aocl_sparse()


