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






def plot_rosko_vs_intel_bar_load(fname = 'rosko_vs_intel_bar_load'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','k','r','r']
	df1 = pandas.read_csv('result_sp1')
	files = df1[df1['algo']=='rosko']['file']._values
	rosko = df1[df1['algo']=='rosko']['time']._values
	rosko_reorder = df1[df1['algo']=='rosko row reorder']['time']._values
	a = rosko / rosko_reorder
	b = [files[i]  for i in range(len(files)) if a[i] > 1.15]
	d = [a[i]  for i in range(len(files)) if a[i] > 1.15]
	labels = [files[i][5:-4] for i in  range(len(files)) if a[i] > 1.15]
	X = np.arange(len(labels))
	plt.figure(figsize = (6,5))
	plt.title('(a) Effect of Row-reordering\non SpMM Throughput', fontsize = 24)
	plt.bar(X + 0.00, [1]*len(d), color = 'g', width = 0.25)
	plt.bar(X + 0.25, d, color = 'r', width = 0.25)
	# plt.bar(X + 0.5, trosko_reorder, color = 'r', width = 0.25)
	plt.xticks(X, labels, fontsize = 18)
	plt.xticks(rotation=60)
	plt.ylabel("Tput relative to Rosko", fontsize = 17)
	# plt.yticks(np.arange(0, 5, 1), fontsize = 16)
	plt.legend(labels=['TUMMY', 'TUMMY+row_reorder'], loc = 'lower right', prop={'size': 16})
	plt.tight_layout()
	plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_rosko_vs_intel_bar_load()


def plot_rosko_vs_intel_load(fname = 'rosko_vs_intel_load'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','k','r','r']
	labels = ['80% Pruning','95% Pruning', 'Ideal Speedup']
	sparsity = [80,95]
	p = range(1,11)
	df1 = pandas.read_csv('result_load')
	#
	plt.figure(figsize = (6,5))
	plt.plot(p, p, label = labels[-1], color = colors[2], linestyle = 'dashed', linewidth=4.0)
	for i in range(len(sparsity)):
		single_core = df1[(df1['sp'] == sparsity[i]) & (df1['p'] == 1)]['time'].mean()
		plt.plot(p, single_core / df1[(df1['sp'] == sparsity[i])]['time']._values, label = labels[i], marker = markers[i], color = colors[3])
	df2 = pandas.read_csv('load_balance')
	mats = ['data/c-36.mtx', 'data/exdata_1.mtx', 
	'data/g7jac020sc.mtx', 'data/r05.mtx', 
	'data/p05.mtx', 'data/r05.mtx', 'data/rdb5000.mtx', 
	'data/s1rmq4m1.mtx', 'data/s3rmt3m3.mtx']
	# labels = [i[:-4] for i in df2[df2['algo'] == 'aocl']['file']._values]
	ext1_before=[];r05_before=[];ext1_after=[];r05_after=[]
	p = range(1,11)
	for i in p:
		ext1_before.append(df2[(df2['algo'] == 'rosko') & (df2['p'] == i) & (df2['file'] == mats[1])]['time']._values[0]) 
		ext1_after.append(df2[(df2['algo'] == 'rosko row reorder') & (df2['p'] == i) & (df2['file'] == mats[1])]['time']._values[0]) 
		r05_before.append(df2[(df2['algo'] == 'rosko') & (df2['p'] == i) & (df2['file'] == mats[-4])]['time']._values[0]) 
		r05_after.append(df2[(df2['algo'] == 'rosko row reorder') & (df2['p'] == i) & (df2['file'] == mats[-4])]['time']._values[0]) 
	labels = ['exdata_1', 'exdata_1+row_reorder', 'r05', 'r05+row_reorder']
	plt.plot(p, [ext1_before[0] / i for i in ext1_before], label = labels[0], marker = markers[0], color = colors[0])
	plt.plot(p, [ext1_after[0] / i for i in ext1_after], label = labels[1], marker = markers[1], color = colors[0])
	plt.plot(p, [r05_before[0] / i for i in r05_before], label = labels[2], marker = markers[0], color = colors[1])
	plt.plot(p, [r05_after[0] / i for i in r05_after], label = labels[3], marker = markers[1], color = colors[1])
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(b) Speedup in Throughput', fontsize = 24)
	plt.xlabel("number of cores", fontsize = 24)
	plt.ylabel("Speedup", fontsize = 24)
	plt.xticks(p, fontsize = 18)
	plt.yticks( fontsize = 20)
	plt.legend(loc = "upper left", prop={'size': 12})
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_rosko_vs_intel_load()




# def plot_rosko_vs_intel_load(fname = 'rosko_vs_intel_load'):
# 	plt.rcParams.update({'font.size': 12})
# 	markers = ['o','v','s','d','^']
# 	colors = ['b','g','k','r','r']
# 	labels = ['80%','87%','95%', 'Ideal']
# 	sparsity = [80,87,95]
# 	p = range(1,11)
# 	df1 = pandas.read_csv('result_load')
# 	#
# 	plt.figure(figsize = (6,4))
# 	plt.plot(p, p, label = labels[-1], color = colors[2], linestyle = 'dashed', linewidth=5.0)
# 	for i in range(len(sparsity)):
# 		single_core = df1[(df1['sp'] == sparsity[i]) & (df1['p'] == 1)]['time'].mean()
# 		plt.plot(p, single_core / df1[(df1['sp'] == sparsity[i])]['time']._values, label = labels[i], marker = markers[i], color = colors[3])
# 	#
# 	#
# 	plt.ticklabel_format(useOffset=False, style='plain')
# 	plt.title('(a) Speedup in Throughput', fontsize = 24)
# 	plt.xlabel("number of cores", fontsize = 24)
# 	plt.ylabel("Speedup", fontsize = 24)
# 	plt.xticks(p, fontsize = 18)
# 	plt.yticks( fontsize = 20)
# 	plt.legend(loc = "upper left", prop={'size': 16})
# 	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
# 	plt.show()
# 	plt.clf()
# 	plt.close('all')
# 	#
