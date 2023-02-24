import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys




def plot_rosko_arm(fname = 'arm_training', ntrials=1):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Rosko_opt', 'Rosko','ARMPL']
	df1 = pandas.read_csv('results_arm_train')
	barWidth = 0.2
	sps = [70,80,90,95,98]
	for model in ['Yolo', 'Resnet', 'Mobnet']:
		rosko = []
		dense = []
		rosko_opt = []
		for sp in sps:
			rosko.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'rosko') & (df1['sp'] == sp)]['time']) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'pack')  & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			dense.append(3*sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			rosko_opt.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'rosko') & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
		br1 = np.arange(len(rosko))
		br2 = [x + barWidth for x in br1]
		br3 = [x + barWidth for x in br2]
		plt.figure(figsize = (12,5))
		plt.bar(br1, rosko_opt, color ='r', width = barWidth,
		        edgecolor ='grey', label = labels[0])
		plt.bar(br2, rosko, color ='m', width = barWidth,
		        edgecolor ='grey', label =labels[1])
		plt.bar(br3, dense, color ='g', width = barWidth,
		        edgecolor ='grey', label =labels[2])
		plt.title('%s Training Runtime For Different Sparsities' % model, fontsize = 24)
		plt.ylabel('Runtime (ssec)', fontsize = 24)
		plt.ylabel('Runtime (ssec)', fontsize = 24)
		plt.xticks([r + barWidth for r in range(len(sps))],
		        sps, rotation = 60, fontsize = 24)
		plt.yticks(fontsize = 24)
		plt.legend(loc = "lower left", fontsize = 20)
		plt.savefig("%s_%s.pdf" % (fname,model), bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')




plot_rosko_arm()





def plot_rosko_arm_full(fname = 'arm_training_full', ntrials=1):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Rosko_opt', 'Rosko','ARMPL']
	df1 = pandas.read_csv('results_arm_train')
	barWidth = 0.2
	sps = [70,80,90,95]
	rosko_full = []; rosko_opt_full = []; dense_full = []
	models = ['Resnet', 'Mobnet']
	for model in models:
		rosko = []
		dense = []
		rosko_opt = []
		for sp in sps:
			rosko.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'rosko') & (df1['sp'] == sp)]['time']) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'pack')  & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			dense.append(3*sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			rosko_opt.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'rosko') & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
		rosko_full.append(70*dense[0] + sum(rosko) + 100*rosko[-1])
		rosko_opt_full.append(70*dense[0] + sum(rosko_opt) + 100*rosko_opt[-1])
		dense_full.append(200*dense[0])
	rosko_full = [dense_full[i] / rosko_full[i] for i in range(len(dense_full))]
	rosko_opt_full = [dense_full[i] / rosko_opt_full[i] for i in range(len(dense_full))]
	dense_full = [dense_full[i] / dense_full[i] for i in range(len(dense_full))]
	br1 = np.arange(len(rosko_full))
	br2 = [x + barWidth for x in br1]
	br3 = [x + barWidth for x in br2]
	plt.figure(figsize = (12,5))
	plt.bar(br1, rosko_opt_full, color ='r', width = barWidth,
	        edgecolor ='grey', label = labels[0])
	plt.bar(br2, rosko_full, color ='m', width = barWidth,
	        edgecolor ='grey', label =labels[1])
	plt.bar(br3, dense_full, color ='g', width = barWidth,
	        edgecolor ='grey', label =labels[2])
	plt.title('Rosko Speedup in End-to-End Training Time', fontsize = 24)
	plt.ylabel('Speedup w.r.t Dense', fontsize = 24)
	plt.xticks([r + barWidth for r in range(len(models))],
	        models, rotation = 60, fontsize = 24)
	plt.yticks(fontsize = 24)
	plt.legend(loc = "lower left", fontsize = 20)
	plt.savefig("%s.pdf" % (fname), bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



plot_rosko_arm_full()






