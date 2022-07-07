import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker



intel_color = '#0071c5'




def plot_rosko_vs_mkl_sp(fname = 'rosko_vs_mkl_sp'):
	plt.rcParams.update({'font.size': 12})
	# all matrices used are 99.87-99.97% sparse
	# labels = ['Fash_mnist', \
	# 'har','indianpines','J_VowelsSmall', \
	# 'kmnist','mnist_test','optdigits',\
	# 'usps','worms20']
	df1 = pandas.read_csv('result_sp')
	labels = [i[5:-4] for i in df1[df1['algo'] == 'rosko']['file']._values]
	labels[0] = 'Fashion_mnist'
	rel_tput = df1[df1['algo'] == 'rosko']['time']._values / df1[df1['algo'] == 'mkl']['time']._values
	X = np.arange(len(labels))
	#
	plt.figure(figsize = (6,5))
	plt.title('(a) Throughput of SpMM in Rosko\nvs MKL', fontsize = 18)
	plt.bar(X + 0.00, rel_tput, color = 'g', width = 0.25)
	plt.bar(X + 0.25, len(labels)*[1], color = 'r', width = 0.25)
	plt.xticks(X, labels, fontsize = 14)
	plt.xticks(rotation=60)
	plt.ylabel("Tput relative to MKL", fontsize = 16)
	# plt.yticks(np.arange(0, 5, 1), fontsize = 16)
	plt.legend(loc = "lower right", labels=['MKL', 'Rosko'])
	plt.tight_layout()
	plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


# plot_rosko_vs_mkl_sp()


def rosko_vs_mkl_suitesparse(fname = 'rosko_vs_mkl_suite'):
	plt.rcParams.update({'font.size': 12})
	df1 = pandas.read_csv('result_sp')
	labels = ['TUMMY Wins','MKL Wins']
	rel_tput = df1[df1['algo'] == 'mkl']['time']._values / df1[df1['algo'] == 'rosko']['time']._values
	sparsity = df1[df1['algo'] == 'mkl']['sparsity']._values
	rel_tput1=[]
	sparsity1=[]
	for i in xrange(len(rel_tput)):
		if sparsity[i] <= 1 and sparsity[i] >= 0.97:
			sparsity1.append(sparsity[i])
			if rel_tput[i] < 1.0:
				rel_tput1.append(-1.0/rel_tput[i])
			else:
				rel_tput1.append(rel_tput[i])
	sparsity1 = [i*100 for i in sparsity1]
	sp_rosko=[]; sp_mkl=[]; rosko_tput=[]; mkl_tput=[]
	for i in range(len(rel_tput1)):
		if rel_tput1[i] > 0:
			rosko_tput.append(rel_tput1[i])
			sp_rosko.append(sparsity1[i])
		else:
			mkl_tput.append(rel_tput1[i])
			sp_mkl.append(sparsity1[i])
	plt.figure(figsize = (6,5))
	plt.title('(a) TUMMY vs MKL SpMM Throughput\nfor SuiteSparse Matrices', fontsize = 18)
	plt.scatter(sp_rosko, rosko_tput, color = 'r', label = labels[0])
	plt.scatter(sp_mkl, mkl_tput, color = intel_color, label = labels[1])
	plt.legend(loc = "lower left", labels=labels)
	plt.plot([0,100], [1,1], color = 'k')
	plt.plot([0,100], [-1,-1], color = 'k')
	plt.xticks(np.arange(98,101,0.5), fontsize = 14)
	plt.yticks([-4, -2, -1, 1, 2, 4, 6, 8], fontsize = 14)
	plt.xlim(98, 100.1)
	plt.xlabel("Sparsity (%)", fontsize = 16)
	plt.ylabel("TUMMY Tput Relative to MKL", fontsize = 16)
	plt.tight_layout()
	plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	files = list(set(df1.file))
	rosko_bw = []
	mkl_bw = []
	sparsity = []
	for i in files:
		s = df1[(df1['algo'] == 'mkl') & (df1['file'] == i)]['sparsity']._values[0]
		if s <= 0.999 and s >= 0.98:
			sparsity.append(s)
			df2 = pandas.read_csv('reports/report_rosko_%s-10.csv' % i[5:],skiprows=17,skipfooter=16)
			rosko_bw.append(float(df2['Average']._values[0]))
			#
			df2 = pandas.read_csv('reports/report_mkl_%s-10.csv' % i[5:],skiprows=17,skipfooter=16)
			mkl_bw.append(float(df2['Average']._values[0]))
	sparsity1 = [i*100 for i in sparsity]
	labels = ['TUMMY','MKL']
	plt.figure(figsize = (6,5))
	plt.title('(b) TUMMY vs MKL SpMM DRAM BW\nfor SuiteSparse Matrices', fontsize = 18)
	plt.scatter(sparsity1, rosko_bw, color = 'r', label = labels[0])
	plt.scatter(sparsity1, mkl_bw, color = intel_color, label = labels[1])
	plt.legend(loc = "lower left", labels=labels)
	plt.xticks(np.arange(98,101,0.5), fontsize = 14)
	plt.xlim(98, 100)
	plt.ylim(0, 18)
	plt.xlabel("Sparsity (%)", fontsize = 16)
	plt.ylabel("DRAM Bandwidth (GB/sec)", fontsize = 16)
	plt.tight_layout()
	plt.savefig("%s_dram.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')

rosko_vs_mkl_suitesparse()



