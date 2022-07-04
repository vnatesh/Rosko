import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker





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
	labels = [i[5:-4] for i in df1[df1['algo'] == 'rosko']['file']._values]
	rel_tput = df1[df1['algo'] == 'mkl']['time']._values / df1[df1['algo'] == 'rosko']['time']._values
	X = np.arange(len(labels))
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
	plt.figure(figsize = (6,5))
	plt.title('(a) Rosko vs MKL SpMM Throughput\nfor SuiteSparse Matrices', fontsize = 18)
	plt.scatter(sparsity1, rel_tput1)
	plt.plot([0,1], [0,0], color = 'r')
	plt.xticks(np.arange(0.98,1.0,0.005), fontsize = 14)
	plt.xlim(0.98, 1.001)
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
	for i in files:
		df1 = pandas.read_csv('reports/report_rosko_%s-10.csv' % i,skiprows=17,skipfooter=16)
		rosko_bw.append(float(df1['Average']._values[0]))
		#
		df1 = pandas.read_csv('reports/report_mkl_%s-10.csv' % i,skiprows=17,skipfooter=16)
		mkl_bw.append(float(df1['Average']._values[0]))


rosko_vs_mkl_suitesparse()



