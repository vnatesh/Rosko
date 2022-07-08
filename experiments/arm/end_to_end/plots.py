import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
from matplotlib import ticker as mticker




def plot_rosko_vs_arm_end_to_end(fname = 'rosko_vs_arm_end_to_end'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['TUMMY', 'CAKE', 'ARMPL','ARMCL']
	t_armpl=[];t_rosko=[];t_armcl=[];t_cake=[];
	#
	df1 = pandas.read_csv('result_end_to_end1')
	#
	df1["sparsity"] = (1.0 - (df1["nz"] / (df1["M"]*df1["K"])))*100
	sparsity = sorted(set(df1["sparsity"]))
	plt.figure(figsize = (6,4))
	for i in sparsity:
		nz = df1[(df1['sparsity'] == i)]['nz']
		N = df1[(df1['sparsity'] == i)]['N']
		flops = nz*N
		t_rosko.append(sum(df1[(df1['algo'] == 'rosko') & (df1['sparsity'] == i)]['time']._values))
		t_cake.append(sum(df1[(df1['algo'] == 'cake') & (df1['sparsity'] == i)]['time']._values))
		t_armpl.append(sum(df1[(df1['algo'] == 'armpl') & (df1['sparsity'] == i)]['time']._values))
		t_armcl.append(sum(df1[(df1['algo'] == 'armcl') & (df1['sparsity'] == i)]['time']._values))
		# gflops_rosko.append((sum(flops) / sum(t_rosko)) / 1e9)
		# gflops_cake.append((sum(flops) / sum(t_cake)) / 1e9)
		# gflops_armpl.append((sum(flops) / sum(t_armpl)) / 1e9)
		# gflops_armcl.append((sum(flops) / sum(t_armcl)) / 1e9)
	#
	plt.plot(sparsity, t_rosko, label = labels[0],  marker = markers[0], color = colors[5])
	plt.plot(sparsity, t_cake, label = labels[1],  marker = markers[2], color = colors[1])
	plt.plot(sparsity, t_armpl, label = labels[2],  marker = markers[1], color = colors[4])
	plt.plot(sparsity, t_armcl, label = labels[3],  marker = markers[3], color = colors[3])
	#
	plt.title('ARM CPU ResNet-50 Inference Runtime', fontsize = 18)
	plt.xlabel("Sparsity (%)", fontsize = 20)
	plt.ylabel("Runtime (sec)", fontsize = 20)
	plt.yticks(np.arange(0,0.31,0.05))
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	


plot_rosko_vs_arm_end_to_end()
