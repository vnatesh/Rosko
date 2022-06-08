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
	rel_tput = df1[df1['algo'] == 'aocl']['time']._values / df1[df1['algo'] == 'rosko']['time']._values
	rel_tput = rel_tput[1:]
	X = np.arange(len(labels))
	#
	plt.figure(figsize = (6,5))
	plt.title('(b) Throughput of SpMM in Rosko vs aocl', fontsize = 18)
	plt.bar(X + 0.00, rel_tput, color = 'r', width = 0.25)
	plt.bar(X + 0.25, 9*[1], color = intel_color, width = 0.25)
	plt.xticks(X, labels, fontsize = 18)
	plt.xticks(rotation=60)
	plt.ylabel("Tput relative to aocl", fontsize = 16)
	plt.yticks(np.arange(0, 5, 1), fontsize = 16)
	plt.legend(labels=['Rosko', 'aocl'])
	plt.tight_layout()
	# plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_rosko_vs_aocl_sparse()

