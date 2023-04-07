import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys


def flops_svd1(M,K,N,d,s):
	return M*K*(d*s*K + N)

def flops_svd2(M,K,N,d,s):
	return s*K*N*(d*M + K)

def flops_rosko(M,K,N,d,s):
	return d*M*K*N

def flops_dense(M,K,N,d,s):
	return d*M*K*N


def plot_rosko_vs_svd(fname = 'rosko_vs_svd'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','m','r','k','m','r']
	labels = ['CAKE', 'SVD+Rosko','Rosko']
	df = pandas.read_csv('result_svd')
	barWidth = 0.2
	sps = [70,80,90,95,98]
	models = ['Yolo', 'Resnet', 'Mobilenet']
	svd = 0.15
	for i in models:
		rosko_tot=[];svd_tot=[];cake_tot=[]
		for j in sps:
			svd_time = []
			svd_dram = []
			df1 = df[(df['layer'].str.contains('%s' % i)) & (df['sp'] == j)]
			layers = df1[df1['algo'] == 'rosko' ]['layer'].values
			rosko_dram = df1[(df1['algo'] == 'rosko_dram')]['time'].values
			cake_dram = df1[(df1['algo'] == 'cake_dram')]['time'].values
			svd_1_dram = df1[(df1['algo'] == 'svd_1_dram')]['time'].values
			svd_2_dram = df1[(df1['algo'] == 'svd_2_dram')]['time'].values
			#
			rosko_time = df1[(df1['algo'] == 'rosko')]['time'].values
			cake_time = df1[(df1['algo'] == 'cake')]['time'].values
			svd_1_time = df1[(df1['algo'] == 'svd_1_rosko')]['time'].values
			svd_2_time = df1[(df1['algo'] == 'svd_2_rosko')]['time'].values
			# svd_time = np.minimum(svd_1_time,svd_2_time)
			for q in range(len(svd_1_time)):
				if svd_1_time[q] < svd_2_time[q]:
					svd_time.append(svd_1_time[q])
					svd_dram.append(svd_1_dram[q])
				else:
					svd_time.append(svd_2_time[q])
					svd_dram.append(svd_2_dram[q])					
			#
			M = df1[(df1['algo'] == 'rosko')]['M'].values
			K = df1[(df1['algo'] == 'rosko')]['K'].values
			N = df1[(df1['algo'] == 'rosko')]['N'].values
			d = 1 - (float(j) / 100.0)
			rosko_flops = np.array(flops_rosko(M, K, N, d, svd)) / rosko_time
			svd_flops = np.array(flops_rosko(M, K, N, d, svd)) / np.array(svd_time)
			cake_flops = np.array(flops_rosko(M, K, N, d, svd)) / cake_time
			#
			rosko_ai = (rosko_flops / rosko_dram) * rosko_time
			svd_ai = (svd_flops / np.array(svd_dram)) * np.array(svd_time)
			cake_ai = (cake_flops / cake_dram) *  cake_time
			#
			rosko_tot.append(sum(rosko_time))
			svd_tot.append(sum(svd_time))
			cake_tot.append(sum(cake_time))
			plt.figure(figsize = (12,5))
			# plt.scatter(layers, cake_time, label = labels[0],  marker = markers[0], color = colors[0], s=20)
			# plt.scatter(layers, svd_time, label = labels[1],  marker = markers[1], color = colors[1], s=20)
			# plt.scatter(layers, rosko_time, label = labels[2],  marker = markers[2], color = colors[2], s=20)
			br1 = np.arange(len(layers))
			br2 = [x + barWidth for x in br1]
			br3 = [x + barWidth for x in br2]
			plt.bar(br1, cake_time, color = colors[0], width = barWidth,
			        edgecolor ='grey', label = labels[0])
			plt.bar(br2, svd_time, color = colors[1], width = barWidth,
			        edgecolor ='grey', label =labels[1])
			plt.bar(br3, rosko_time, color = colors[2], width = barWidth,
			        edgecolor ='grey', label =labels[2])
			plt.title('%s Measured Performance at %d%% Sparsity' % (i, j), fontsize = 24)
			plt.ylabel('Runtime (sec)', fontsize = 24)
			# plt.xticks(rotation = 60, fontsize = 16)
			plt.xticks([r + barWidth for r in range(len(layers))],
		        layers, rotation = 60, fontsize = 16)
			plt.yticks(fontsize = 24)
			if i == 'Yolo':
				plt.legend(loc = "upper right", fontsize = 20)
			else:
				plt.legend(loc = "upper left", fontsize = 20)
			plt.savefig("%s_%s_%s.pdf" % (fname,i,j), bbox_inches='tight')
			plt.show()
			plt.clf()
			plt.close('all')
			#
			#
			# plt.figure(figsize = (12,5))
			# # plt.scatter(layers, cake_time, label = labels[0],  marker = markers[0], color = colors[0], s=20)
			# # plt.scatter(layers, svd_time, label = labels[1],  marker = markers[1], color = colors[1], s=20)
			# # plt.scatter(layers, rosko_time, label = labels[2],  marker = markers[2], color = colors[2], s=20)
			# br1 = np.arange(len(layers))
			# br2 = [x + barWidth for x in br1]
			# br3 = [x + barWidth for x in br2]
			# plt.bar(br1, cake_dram, color = colors[0], width = barWidth,
			#         edgecolor ='grey', label = labels[0])
			# plt.bar(br2, svd_dram, color = colors[1], width = barWidth,
			#         edgecolor ='grey', label =labels[1])
			# plt.bar(br3, rosko_dram, color = colors[2], width = barWidth,
			#         edgecolor ='grey', label =labels[2])
			# plt.title('%s Modelled Performance at %d%% Sparsity' % (i, j), fontsize = 24)
			# plt.ylabel('A.I. (FLOPs/byte)', fontsize = 24)
			# # plt.xticks(rotation = 60, fontsize = 16)
			# plt.xticks([r + barWidth for r in range(len(layers))],
		 #        layers, rotation = 60, fontsize = 16)
			# plt.yticks(fontsize = 24)
			# if i == 'Yolo':
			# 	plt.legend(loc = "upper right", fontsize = 20)
			# else:
			# 	plt.legend(loc = "upper left", fontsize = 20)
			# # plt.savefig("%s_%s.pdf" % (fname,j), bbox_inches='tight')
			# plt.show()
			# plt.clf()
			# plt.close('all')
		plt.figure(figsize = (12,5))
		# plt.scatter(layers, cake_time, label = labels[0],  marker = markers[0], color = colors[0], s=20)
		# plt.scatter(layers, svd_time, label = labels[1],  marker = markers[1], color = colors[1], s=20)
		# plt.scatter(layers, rosko_time, label = labels[2],  marker = markers[2], color = colors[2], s=20)
		br1 = np.arange(len(sps))
		br2 = [x + barWidth for x in br1]
		br3 = [x + barWidth for x in br2]
		plt.bar(br1, cake_tot, color = colors[0], width = barWidth,
		        edgecolor ='grey', label = labels[0])
		plt.bar(br2, svd_tot, color = colors[1], width = barWidth,
		        edgecolor ='grey', label =labels[1])
		plt.bar(br3, rosko_tot, color = colors[2], width = barWidth,
		        edgecolor ='grey', label =labels[2])
		plt.title('%s Forward Pass Performance' % (i), fontsize = 24)
		plt.ylabel('Runtime (sec)', fontsize = 24)
		plt.xlabel('Sparsity (%)', fontsize = 24)
		# plt.xticks(rotation = 60, fontsize = 16)
		plt.xticks([r + barWidth for r in range(len(sps))],
	        sps, rotation = 0, fontsize = 16)
		plt.yticks(fontsize = 24)
		plt.legend(loc = "lower left", fontsize = 20)
		plt.savefig("%s_%s_forward.pdf" % (fname,i), bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')


plot_rosko_vs_svd()



def plot_rosko_arm(fname = 'arm_training', ntrials=1):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Rosko_opt', 'Rosko','ARMPL', 'SVD+Rosko']
	df1 = pandas.read_csv('results_arm_train')
	barWidth = 0.2
	sps = [70,80,90,95,98]
	for model in ['Yolo', 'Resnet', 'Mobilenet']:
		rosko = []
		dense = []
		rosko_opt = []
		svd = []
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
		br4 = [x + barWidth for x in br3]
		plt.figure(figsize = (12,5))
		plt.bar(br1, rosko_opt, color ='r', width = barWidth,
		        edgecolor ='grey', label = labels[0])
		plt.bar(br2, rosko, color ='m', width = barWidth,
		        edgecolor ='grey', label =labels[1])
		plt.bar(br3, dense, color ='g', width = barWidth,
		        edgecolor ='grey', label =labels[2])
		plt.bar(br4, svd, color ='k', width = barWidth,
		        edgecolor ='grey', label =labels[3])
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
	df1 = pandas.read_csv('results_arm_train_1')
	barWidth = 0.2
	sps = [70,80,90,95,98]
	rosko_full = []; rosko_opt_full = []; dense_full = []
	models = ['ResNet-18', 'MobileNetV2']
	models1 = ['ResNet-18', 'MobileNetV2', 'VGG-19']
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
	#
	dense_full.append(dense_full[-1])
	rosko_opt_full.append(rosko_opt_full[-1])
	# rosko_full = [dense_full[i] / rosko_full[i] for i in range(len(dense_full))]
	rosko_opt_full = [dense_full[i] / rosko_opt_full[i] for i in range(len(dense_full))]
	dense_full = [dense_full[i] / dense_full[i] for i in range(len(dense_full))]
	br1 = np.arange(len(dense_full))
	br2 = [x + barWidth for x in br1]
	plt.figure(figsize = (12,5))
	plt.bar(br1, rosko_opt_full, color ='r', width = barWidth,
	        edgecolor ='grey', label = labels[1])
	plt.bar(br2, dense_full, color ='m', width = barWidth,
	        edgecolor ='grey', label =labels[2])
	plt.title('Rosko Speedup in End-to-End Training Time', fontsize = 24)
	plt.ylabel('Speedup w.r.t Dense', fontsize = 24)
	plt.xticks([r + barWidth for r in range(len(models1))],
	        models1, rotation = 0, fontsize = 24)
	plt.yticks(fontsize = 24)
	plt.legend(loc = "lower left", fontsize = 20)
	plt.savefig("%s.pdf" % (fname), bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



plot_rosko_arm_full()





def plot_rosko_arm(fname = 'arm_training', ntrials=1):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r']
	labels = ['Rosko_opt', 'Rosko','ARMPL', 'SVD+Rosko1', 'SVD+Rosko2']
	df1 = pandas.read_csv('result_svd')
	barWidth = 0.15
	sps = [90,95,98]
	for model in ['Yolo', 'Resnet', 'Mobilenet']:
		rosko = []
		dense = []
		rosko_opt = []
		svd1 = []
		svd2 = []
		for sp in sps:
			rosko.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'rosko') & (df1['sp'] == sp)]['time']) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'pack')  & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			dense.append(3*sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			rosko_opt.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'rosko') & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			svd1.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'svd_1_rosko') & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
			svd2.append(2*(sum(df1[(df1['layer'] == model) & (df1['algo'] == 'svd_2_rosko') & (df1['sp'] == sp)]['time'])) \
			+ sum(df1[(df1['layer'] == model) & (df1['algo'] == 'dense') & (df1['sp'] == sp)]['time']))
		br1 = np.arange(len(rosko))
		br2 = [x + barWidth for x in br1]
		br3 = [x + barWidth for x in br2]
		br4 = [x + barWidth for x in br3]
		br5 = [x + barWidth for x in br4]
		plt.figure(figsize = (12,5))
		plt.bar(br1, rosko_opt, color ='r', width = barWidth,
		        edgecolor ='grey', label = labels[0])
		plt.bar(br2, rosko, color ='m', width = barWidth,
		        edgecolor ='grey', label =labels[1])
		plt.bar(br3, dense, color ='g', width = barWidth,
		        edgecolor ='grey', label =labels[2])
		plt.bar(br4, svd1, color ='k', width = barWidth,
		        edgecolor ='grey', label =labels[3])
		plt.bar(br5, svd2, color ='b', width = barWidth,
		        edgecolor ='grey', label =labels[4])
		plt.title('%s Training Runtime For Different Sparsities' % model, fontsize = 24)
		plt.ylabel('Runtime (ssec)', fontsize = 24)
		plt.xticks([r + barWidth for r in range(len(sps))],
		        sps, rotation = 60, fontsize = 24)
		plt.yticks(fontsize = 24)
		plt.legend(loc = "lower left", fontsize = 20)
		# plt.savefig("%s_%s.pdf" % (fname,model), bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')




plot_rosko_arm()

