import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import numpy as np
import os
import re
import sys
import itertools
from matplotlib import ticker as mticker





def plot_rosko_vs_intel_test(fname = 'rosko_vs_intel_test', ntrials = 10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'b', 'g', 'k','g','m']
	barWidth = 0.15
	#
	df1 = pandas.read_csv('results_intel_test1')
	# sps = [80, 85, 90, 95, 98]
	sps = [80, 90, 98]
	labels1 = ['Rosko Measured','Rosko Modelled', 'MKL']
	labels2 = ['%d%%' % i for i in sps] + ['dense']
	layers = df1[(df1['algo'] == 'rosko') & (df1['sp'] == sps[0])]['layer']._values
	cores = range(2,11)
	error = 0
	#
	rosko_times = []
	rosko_dram = []
	rosko_pred_times = []
	rosko_pred_dram = []
	arm_times = []
	arm_dram = []
	for sp in sps:
		gflops_armpl=[];gflops_rosko=[];dram_io_rosko=[];dram_io_armpl=[];
		gflops_armcl=[]; dram_io_armcl=[]; dram_bw_rosko=[];dram_bw_armpl=[];
		dram_bw_armcl=[];time_armcl=[]; time_armpl=[]; time_rosko=[];time_rosko1=[]; flops = []
		for p in cores:
			N = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['N']._values[0]
			M = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['M']._values[0]
			K = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['K']._values[0]
			nz = (1.0 - (float(sp) / 100.0))*M*K
			flops.append(nz) 
			df2 = pandas.read_csv('reports_intel_test1/report_rosko_%d-%d.csv' % (p,sp) ,skiprows=17,skipfooter=16)
			cpu_time = df1[(df1['algo'] == 'rosko') & (df1['p'] == p) & (df1['sp'] == sp)]['time']._values[0]
			dram_bw = df2['Average']._values[0]
			gflops_rosko.append((nz*N) / cpu_time / 1e9)
			df2 = pandas.read_csv('reports_intel_test1/report_rosko_%d-%d.csv' % (p,sp),skipfooter=20)
			elapsed = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0]
			#
			df3 = pandas.read_csv('reports_intel_test1/report_setup_%d-%d.csv' % (p,sp) ,skiprows=17,skipfooter=16)
			dram_bw1 = df3['Average']._values[0]
			df3 = pandas.read_csv('reports_intel_test1/report_setup_%d-%d.csv' % (p,sp),skipfooter=20)
			elapsed1 = df3[df3['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0]
			dram_bw = ((dram_bw*elapsed) - (dram_bw1*elapsed1)) / (elapsed-elapsed1)
			dram_bw_rosko.append(dram_bw)
			time_rosko.append((elapsed-elapsed1) / ntrials)
		dram_io_rosko_pred = df1[(df1['algo'] == 'rosko_dram') & (df1['sp'] == sp) & (df1['p'].isin(cores))]['time']._values
		dram_bw_rosko_pred = list(dram_io_rosko_pred / np.array(time_rosko))
		error += np.mean(dram_bw_rosko_pred / np.array(dram_bw_rosko))
		rosko_pred_dram.append(dram_bw_rosko_pred)
		rosko_pred_times.append(time_rosko)
		rosko_dram.append(dram_bw_rosko)
		print(np.mean(dram_bw_rosko_pred / np.array(dram_bw_rosko)))
		if sp == 98:
			print((dram_bw_rosko_pred, dram_bw_rosko))
		# arm_dram.append(dram_bw_armpl)
	#
	#
	#
	plot_lines = []
	plt.figure(figsize = (6,4))
	plt.title('(c) Rosko DRAM Bandwidth\nUsage on Intel i9 CPU', fontsize = 20)
	for i in range(len(sps)):
		l1, = plt.plot(cores, rosko_dram[i], color = colors[i])
		l2, = plt.plot(cores, rosko_pred_dram[i], color = colors[i], linestyle = 'dashed')
		plot_lines.append([l1, l2])
	legend1 = plt.legend(plot_lines[0], labels1, loc='upper left', prop={'size': 12})
	plt.legend([l[0] for l in plot_lines], labels2, loc='lower right', prop={'size': 12})
	plt.gca().add_artist(legend1)
	plt.xlabel("Number of Cores", fontsize = 20)
	plt.xticks(cores)
	plt.yticks(fontsize = 16)
	plt.ylim(0,max(list(itertools.chain.from_iterable(rosko_dram+rosko_pred_dram)))*1.05)
	plt.ylabel("DRAM BW (GB/sec)", fontsize = 20)
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	print(error / len(sps))



plot_rosko_vs_intel_test()






def plot_rosko_vs_arm_test(fname = 'rosko_vs_arm_test', ntrials = 1):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'b', 'g', 'k','g','m']
	barWidth = 0.15
	#
	df1 = pandas.read_csv('results_arm_test')
	# sps = [80, 85, 90, 95, 98]
	sps = [80, 90, 98]
	labels1 = ['Rosko Measured','Rosko Modelled', 'ARMPL']
	labels2 = ['%d%%' % i for i in sps] + ['dense']
	layers = df1[(df1['algo'] == 'rosko') & (df1['sp'] == sps[0])]['layer']._values
	cores = range(1,5)
	error = 0
	#
	rosko_times = []
	rosko_dram = []
	rosko_pred_times = []
	rosko_pred_dram = []
	arm_times = []
	arm_dram = []
	for sp in sps:
		gflops_armpl=[];gflops_rosko=[];dram_io_rosko=[];dram_io_armpl=[];
		gflops_armcl=[]; dram_io_armcl=[]; dram_bw_rosko=[];dram_bw_armpl=[];
		dram_bw_armcl=[];time_armcl=[]; time_armpl=[]; time_rosko=[];time_rosko1=[]; flops = []
		for p in cores:
			N = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['N']._values[0]
			M = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['M']._values[0]
			K = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['K']._values[0]
			nz = (1.0 - (float(sp) / 100.0))*M*K
			flops.append(nz) 
			a = open('reports_arm_test/report_rosko_%s-%d' % (p,sp),'r').read().split('\n')
			b = open('reports_arm_test/report_setup_%s-%d' % (p,sp),'r').read().split('\n')
			cpu_time = df1[(df1['algo'] == 'rosko') & (df1['p'] == p) & (df1['sp'] == sp)]['time']._values[0]
			tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
			tmp -= (((int(re.search(r'\d+', b[5]).group())*64.0))) / 1e9
			#
			# tmp = (((int(re.search(r'\d+', a[9]).group())*4.0))) / 1e9
			#
			# tmp = (((int(re.search(r'\d+', a[7]).group())*4.0))) / 1e9
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9
			# tmp -= (((int(re.search(r'\d+', b[9]).group())*16.0))) / 1e9
			# tmp = (((int(re.search(r'\d+', a[7]).group())*4.0))) / 1e9
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9			
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9
			# tmp -= 2*(nz + K*N)*4 / 1e9
			elapsed = float(re.search(r'\d+\.\d+', a[11]).group())
			elapsed -= float(re.search(r'\d+\.\d+', b[11]).group())
			# elapsed -= df1[(df1['algo'] == 'setup') & (df1['layer'] == layer) & (df1['sp'] == sp)]['time']._values[0]
			# elapsed += float(re.search(r'\d+\.\d+', a[-4]).group())
			dram_io_rosko.append(tmp / ntrials)
			# dram_bw_rosko.append(tmp / elapsed)
			dram_bw_rosko.append(tmp / ntrials / cpu_time)
			# gflops_rosko.append((nz*N) / cpu_time / 1e9)
			gflops_rosko.append((M*K*N) / cpu_time / 1e9)
			time_rosko.append(cpu_time)
			print("elapsed time = %f" % (elapsed / ntrials))
			print("cpu time = %f" % (cpu_time))
			print
			a = open('reports_arm_test/report_armpl_%s-%d' % (p,sp),'r').read().split('\n')
			cpu_time = df1[(df1['algo'] == 'armpl') & (df1['p'] == p) & (df1['sp'] == sp)]['time']._values[0]
			tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
			dram_io_armpl.append(tmp / ntrials)
			dram_bw_armpl.append(tmp / ntrials / cpu_time)
			gflops_armpl.append((M*K*N) / cpu_time / 1e9)
		dram_io_rosko_pred = df1[(df1['algo'] == 'rosko_dram') & (df1['sp'] == sp)]['time']._values
		dram_bw_rosko_pred = list(dram_io_rosko_pred / np.array(time_rosko))
		error += np.mean(dram_bw_rosko_pred / np.array(dram_bw_rosko))
		rosko_pred_dram.append(dram_bw_rosko_pred)
		rosko_pred_times.append(time_rosko)
		rosko_dram.append(dram_bw_rosko)
		arm_dram.append(dram_bw_armpl)
	#
	#
	#
	plot_lines = []
	plt.figure(figsize = (6,4))
	plt.title('(d) Rosko DRAM Bandwidth Usage\non ARM Cortex-A53 CPU', fontsize = 20)
	for i in range(len(sps)):
		l1, = plt.plot(cores, rosko_dram[i], color = colors[i])
		l2, = plt.plot(cores, rosko_pred_dram[i], color = colors[i], linestyle = 'dashed')
		# l3, = plt.plot(cores, arm_dram[-1], marker = markers[1], color = colors[-1])
		# plot_lines.append([l1, l2, l3])
		plot_lines.append([l1, l2])
	legend1 = plt.legend(plot_lines[0], labels1, loc='upper left', prop={'size': 12})
	plt.legend([l[0] for l in plot_lines], labels2, loc='lower right', prop={'size': 12})
	plt.gca().add_artist(legend1)
	plt.xlabel("Number of Cores", fontsize = 20)
	plt.xticks(cores)
	plt.yticks(fontsize = 16)
	# plt.ylim(0,max(list(itertools.chain.from_iterable(rosko_dram+rosko_pred_dram+arm_dram))))
	plt.ylim(0,max(list(itertools.chain.from_iterable(rosko_dram+rosko_pred_dram))))
	plt.ylabel("DRAM BW (GB/sec)", fontsize = 20)
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	print(error / len(sps))



plot_rosko_vs_arm_test()






def plot_rosko_vs_intel_valid(fname = 'rosko_vs_intel_valid', ntrials = 10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'b', 'g', 'k','g','m']
	labels = ['Rosko','MKL-Dense', 'MKL-CSR', 'TACO']
	df1 = pandas.read_csv('results_valid_intel')
	M = df1[(df1['algo'] == 'mkl')]['M']._values[0]
	K = df1[(df1['algo'] == 'mkl')]['K']._values[0]
	sps = [75, 80, 82, 85, 87, 90, 92, 95, 97, 99, 99.9]
	# mkl = [(1.0 - (sps[i] / 100.0))*M*K / df1[(df1['algo'] == 'mkl')]['time']._values[0] for i in range(len(sps))]
	# rosko = [(1.0 - (sps[i] / 100.0))*M*K / df1[(df1['algo'] == 'rosko')]['time']._values[i] for i in range(len(sps))]
	# mkl_csr = [(1.0 - (sps[i] / 100.0))*M*K / df1[(df1['algo'] == 'mkl_csr')]['time']._values[i] for i in range(len(sps))]
	mkl = len(sps)*[df1[(df1['algo'] == 'mkl')]['time']._values[0]]
	rosko = df1[(df1['algo'] == 'rosko')]['time']._values
	mkl_csr = df1[(df1['algo'] == 'mkl_csr')]['time']._values
	taco = df1[(df1['algo'] == 'taco')]['time']._values
	plt.figure(figsize = (6,4))
	plt.title('(a) SpMM Runtime at Various\nSparsities on Intel i9', fontsize = 20)
	plt.plot(sps, rosko[1:], color = colors[0], label = labels[0])
	plt.plot(sps, mkl, color = colors[1], label = labels[1])
	plt.plot(sps, taco[1:], color = colors[2], label = labels[3])
	plt.plot(sps, mkl_csr[1:], color = colors[3], label = labels[2])
	plt.legend(loc = "lower left", prop={'size': 12})
	plt.xlabel("Sparsity (%)", fontsize = 20)
	plt.xticks(fontsize = 16)
	plt.ylim(0,max(mkl_csr[9:]))
	plt.yticks(fontsize = 16)
	plt.ylabel("Runtime (sec)", fontsize = 20)
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



plot_rosko_vs_intel_valid()





def plot_rosko_vs_arm_valid(fname = 'rosko_vs_arm_valid', ntrials = 10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'b', 'g', 'k','g','m']
	labels = ['Rosko','ARMPL', 'TACO']
	df1 = pandas.read_csv('results_valid_arm')
	M = df1[(df1['algo'] == 'armpl')]['M']._values[0]
	K = df1[(df1['algo'] == 'armpl')]['K']._values[0]
	sps = [55, 60, 65, 70, 75, 80, 82, 85, 87, 90, 92, 95, 97, 99, 99.9]
	# mkl = [(1.0 - (sps[i] / 100.0))*M*K / df1[(df1['algo'] == 'mkl')]['time']._values[0] for i in range(len(sps))]
	# rosko = [(1.0 - (sps[i] / 100.0))*M*K / df1[(df1['algo'] == 'rosko')]['time']._values[i] for i in range(len(sps))]
	# mkl_csr = [(1.0 - (sps[i] / 100.0))*M*K / df1[(df1['algo'] == 'mkl_csr')]['time']._values[i] for i in range(len(sps))]
	armpl = len(sps)*[df1[(df1['algo'] == 'armpl')]['time']._values[0]]
	rosko = df1[(df1['algo'] == 'rosko')]['time']._values
	taco = df1[(df1['algo'] == 'taco')]['time']._values
	plt.figure(figsize = (6,4))
	plt.title('(b) SpMM Runtime at Various\nSparsities on ARM Cortex A-53', fontsize = 20)
	plt.plot(sps, rosko, color = colors[0], label = labels[0])
	plt.plot(sps, armpl, color = colors[1], label = labels[1])
	plt.plot(sps, taco, color = colors[2], label = labels[2])
	plt.legend(loc = "lower left", prop={'size': 12})
	plt.xlabel("Sparsity (%)", fontsize = 20)
	plt.xticks(fontsize = 16)
	plt.ylim(0,max(taco[11:]))
	plt.yticks(fontsize = 16)
	plt.ylabel("Runtime (sec)", fontsize = 20)
	plt.savefig("%s.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



plot_rosko_vs_arm_valid()


#-------------------------------------------------------------------------#










def plot_rosko_vs_arm_dnn(fname = 'rosko_vs_arm_dlmc', ntrials = 1):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'm', 'k', 'b','g','aqua']
	labels = ['Rosko', 'ARMPL','ARMCL', 'Rosko_pred']
	#
	df1 = pandas.read_csv('results_arm_train')
	sps = [80, 85, 90, 95, 98]
	layers = df1[(df1['algo'] == 'rosko') & (df1['sp'] == sps[0])]['layer']._values
	cores = range(1,5)
	#
	for sp in sps:
		gflops_armpl=[];gflops_rosko=[];dram_io_rosko=[];dram_io_armpl=[];
		gflops_armcl=[]; dram_io_armcl=[]; dram_bw_rosko=[];dram_bw_armpl=[];
		dram_bw_armcl=[];time_armcl=[]; time_armpl=[]; time_rosko=[];time_rosko1=[]; flops = []
		for layer in layers:
			N = df1[(df1['algo'] == 'rosko') & (df1['layer'] == layer)]['N']._values[0]
			M = df1[(df1['algo'] == 'rosko') & (df1['layer'] == layer)]['M']._values[0]
			K = df1[(df1['algo'] == 'rosko') & (df1['layer'] == layer)]['K']._values[0]
			nz = (1.0 - (float(sp) / 100.0))*M*K
			flops.append(nz) 
			a = open('reports_arm_train/report_rosko_%s-%d' % (layer,sp),'r').read().split('\n')
			b = open('reports_arm_train/report_setup_%s-%d' % (layer,sp),'r').read().split('\n')
			cpu_time = df1[(df1['algo'] == 'rosko') & (df1['layer'] == layer) & (df1['sp'] == sp)]['time']._values[0]
			tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
			tmp -= (((int(re.search(r'\d+', b[5]).group())*64.0))) / 1e9
			#
			# tmp = (((int(re.search(r'\d+', a[9]).group())*4.0))) / 1e9
			#
			# tmp = (((int(re.search(r'\d+', a[7]).group())*4.0))) / 1e9
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9
			# tmp -= (((int(re.search(r'\d+', b[9]).group())*16.0))) / 1e9
			# tmp = (((int(re.search(r'\d+', a[7]).group())*4.0))) / 1e9
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9			
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9
			# tmp -= 2*(nz + K*N)*4 / 1e9
			elapsed = float(re.search(r'\d+\.\d+', a[11]).group())
			elapsed -= float(re.search(r'\d+\.\d+', b[11]).group())
			# elapsed -= df1[(df1['algo'] == 'setup') & (df1['layer'] == layer) & (df1['sp'] == sp)]['time']._values[0]
			# elapsed += float(re.search(r'\d+\.\d+', a[-4]).group())
			dram_io_rosko.append(tmp / ntrials)
			# dram_bw_rosko.append(tmp / elapsed)
			dram_bw_rosko.append(tmp / ntrials / cpu_time)
			gflops_rosko.append((nz*N) / cpu_time / 1e9)
			time_rosko.append(cpu_time)
			print("elapsed time = %f" % (elapsed / ntrials))
			print("cpu time = %f" % (cpu_time))
			print
			# time_rosko.append(elapsed / ntrials)
		dram_io_rosko_pred = df1[(df1['algo'] == 'rosko_dram') & (df1['sp'] == sp)]['time']._values
		dram_bw_rosko_pred = dram_io_rosko_pred / np.array(time_rosko) 
		print(dram_io_rosko)
		print(dram_io_rosko_pred)
		print
		print(dram_io_rosko_pred / np.array(dram_io_rosko))
		print(dram_bw_rosko_pred / np.array(dram_bw_rosko))
		#
		flops = np.log10(np.array(flops))
		plt.figure(figsize = (6,4))
		plt.scatter(flops, gflops_rosko, label = labels[0],  marker = markers[0], color = colors[0], s=20)
		# plt.scatter(flops, gflops_armpl, label = labels[1],  marker = markers[1], color = colors[1], s=20)
		# plt.scatter(flops, gflops_armcl, label = labels[2],  marker = markers[3], color = colors[2], s=20)
		#
		plt.title('(a) Throughput for SpMM in\nCNN Layers', fontsize = 24)
		plt.xlabel("# of nonzeroes (log scale)", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 18)
		plt.legend(loc = "upper left", prop={'size': 12})
		# plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		#
		#
		plt.figure(figsize = (6,4))
		# plt.scatter(flops, dram_io_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.scatter(flops, dram_io_armcl, label = labels[2],  marker = markers[2], color = colors[2])
		plt.scatter(flops, dram_io_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		plt.scatter(flops, dram_io_rosko_pred, label = labels[3],  marker = markers[0], color = colors[3])
		#
		plt.title('(b) DRAM IO for SpMM in\nCNN Layers', fontsize = 24)
		plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylabel("DRAM IO (GB)", fontsize = 24)
		plt.legend(loc = "upper left", prop={'size': 12})
		# plt.savefig("%s_io.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		#
		#
		plt.figure(figsize = (6,4))
		plt.scatter(flops, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		# plt.scatter(flops, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.scatter(flops, dram_bw_armcl, label = labels[2],  marker = markers[2], color = colors[2])
		plt.scatter(flops, dram_bw_rosko_pred, label = labels[3],  marker = markers[0], color = colors[3])
		#
		plt.title('(c) DRAM BW for SpMM in\nCNN Layers', fontsize = 24)
		plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
		plt.legend(loc = "upper right", prop={'size': 12})
		# plt.savefig("%s_bw.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		#
		#
		plt.figure(figsize = (6,4))
		plt.scatter(time_rosko, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		# plt.scatter(time_armpl, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.scatter(time_armcl, dram_bw_armcl, label = labels[2],  marker = markers[2], color = colors[2])
		plt.scatter(time_rosko, dram_bw_rosko_pred, label = labels[3],  marker = markers[0], color = colors[3])
		#
		plt.title('(c) BW Required to Attain\nTarget Runtime', fontsize = 24)
		plt.xlabel("Runtime (sec)", fontsize = 24)
		# plt.xticks(np.arange(0,0.31,0.05), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
		plt.legend(loc = "upper right", prop={'size': 12})
		# plt.savefig("%s_bw_tput.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')

		


plot_rosko_vs_arm_dnn()








def plot_rosko_vs_arm_test(fname = 'rosko_vs_arm_dlmc', ntrials = 1):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'm', 'k', 'b','g','aqua']
	labels = ['Rosko', 'ARMPL','ARMCL', 'Rosko_pred']
	barWidth = 0.15
	#
	df1 = pandas.read_csv('results_arm_test')
	sps = [80, 85, 90, 95, 98]
	layers = df1[(df1['algo'] == 'rosko') & (df1['sp'] == sps[0])]['layer']._values
	cores = range(1,5)
	#
	for sp in sps:
		gflops_armpl=[];gflops_rosko=[];dram_io_rosko=[];dram_io_armpl=[];
		gflops_armcl=[]; dram_io_armcl=[]; dram_bw_rosko=[];dram_bw_armpl=[];
		dram_bw_armcl=[];time_armcl=[]; time_armpl=[]; time_rosko=[];time_rosko1=[]; flops = []
		for p in cores:
			N = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['N']._values[0]
			M = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['M']._values[0]
			K = df1[(df1['algo'] == 'rosko') & (df1['p'] == p)]['K']._values[0]
			nz = (1.0 - (float(sp) / 100.0))*M*K
			flops.append(nz) 
			a = open('reports_arm_test/report_rosko_%s-%d' % (p,sp),'r').read().split('\n')
			b = open('reports_arm_test/report_setup_%s-%d' % (p,sp),'r').read().split('\n')
			cpu_time = df1[(df1['algo'] == 'rosko') & (df1['p'] == p) & (df1['sp'] == sp)]['time']._values[0]
			tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
			tmp -= (((int(re.search(r'\d+', b[5]).group())*64.0))) / 1e9
			#
			# tmp = (((int(re.search(r'\d+', a[9]).group())*4.0))) / 1e9
			#
			# tmp = (((int(re.search(r'\d+', a[7]).group())*4.0))) / 1e9
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9
			# tmp -= (((int(re.search(r'\d+', b[9]).group())*16.0))) / 1e9
			# tmp = (((int(re.search(r'\d+', a[7]).group())*4.0))) / 1e9
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9			
			# tmp += (((int(re.search(r'\d+', a[8]).group())*4.0))) / 1e9
			# tmp -= 2*(nz + K*N)*4 / 1e9
			elapsed = float(re.search(r'\d+\.\d+', a[11]).group())
			elapsed -= float(re.search(r'\d+\.\d+', b[11]).group())
			# elapsed -= df1[(df1['algo'] == 'setup') & (df1['layer'] == layer) & (df1['sp'] == sp)]['time']._values[0]
			# elapsed += float(re.search(r'\d+\.\d+', a[-4]).group())
			dram_io_rosko.append(tmp / ntrials)
			# dram_bw_rosko.append(tmp / elapsed)
			dram_bw_rosko.append(tmp / ntrials / cpu_time)
			# gflops_rosko.append((nz*N) / cpu_time / 1e9)
			gflops_rosko.append((M*K*N) / cpu_time / 1e9)
			time_rosko.append(cpu_time)
			print("elapsed time = %f" % (elapsed / ntrials))
			print("cpu time = %f" % (cpu_time))
			print
			a = open('reports_arm_test/report_armpl_%s-%d' % (p,sp),'r').read().split('\n')
			cpu_time = df1[(df1['algo'] == 'armpl') & (df1['p'] == p) & (df1['sp'] == sp)]['time']._values[0]
			tmp = (((int(re.search(r'\d+', a[5]).group())*64.0))) / 1e9
			dram_io_armpl.append(tmp / ntrials)
			dram_bw_armpl.append(tmp / ntrials / cpu_time)
			gflops_armpl.append((M*K*N) / cpu_time / 1e9)
		dram_io_rosko_pred = df1[(df1['algo'] == 'rosko_dram') & (df1['sp'] == sp)]['time']._values
		dram_bw_rosko_pred = dram_io_rosko_pred / np.array(time_rosko) 
		print(dram_io_rosko)
		print(dram_io_rosko_pred)
		print
		print(dram_io_rosko_pred / np.array(dram_io_rosko))
		print(dram_bw_rosko_pred / np.array(dram_bw_rosko))
		#
		flops = np.log10(np.array(flops))
		plt.figure(figsize = (6,4))
		plt.plot(cores, gflops_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		plt.plot(cores, gflops_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.plot(flops, gflops_armcl, label = labels[2],  marker = markers[3], color = colors[2], s=20)
		#
		plt.title('(a) Throughput for SpMM in\nCNN Layers', fontsize = 24)
		plt.xlabel("# of cores", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylim(0,max(list(gflops_rosko) + list(gflops_armpl))*1.1)
		plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 18)
		plt.legend(loc = "upper left", prop={'size': 12})
		# plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		#
		#
		plt.figure(figsize = (6,4))
		plt.plot(cores, dram_io_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.plot(flops, dram_io_armcl, label = labels[2],  marker = markers[2], color = colors[2])
		plt.plot(cores, dram_io_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		plt.plot(cores, dram_io_rosko_pred, label = labels[3],  marker = markers[0], color = colors[3])
		# br1 = np.arange(len(cores))
		# br2 = [x + barWidth for x in br1]
		# plt.bar(br1, dram_io_rosko, color =colors[0], width = barWidth,
		#         edgecolor ='grey', label = labels[0])
		# plt.bar(br2, dram_io_rosko_pred, color =colors[3], width = barWidth,
		#         edgecolor ='grey', label =labels[3])
		# plt.title('(b) Rosko SpMM DRAM IO\n On ARM Cortex A53', fontsize = 24)
		# plt.xticks([r + barWidth for r in range(len(cores))],
		#         cores, rotation = 60, fontsize = 24)
		plt.xlabel("# of cores", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylim(0,max(list(dram_io_rosko_pred) + list(dram_io_rosko) + list(dram_io_armpl))*1.1)
		plt.ylabel("DRAM IO (GB)", fontsize = 24)
		plt.legend(loc = "upper left", prop={'size': 12})
		# plt.savefig("%s_io.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		#
		#
		plt.figure(figsize = (6,4))
		plt.plot(cores, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		plt.plot(cores, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.plot(flops, dram_bw_armcl, label = labels[2],  marker = markers[2], color = colors[2])
		plt.plot(cores, dram_bw_rosko_pred, label = labels[3],  marker = markers[0], color = colors[3])
		#
		plt.title('(c) DRAM BW for SpMM in\nCNN Layers', fontsize = 24)
		plt.xlabel("# of cores", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylim(0,max(list(dram_bw_rosko_pred) + list(dram_bw_rosko) + list(dram_bw_armpl))*1.1)
		plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
		plt.legend(loc = "upper left", prop={'size': 12})
		# plt.savefig("%s_bw.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')





plot_rosko_vs_arm_test()






def plot_rosko_vs_intel_dnn(fname = 'rosko_vs_arm_dlmc', ntrials = 10):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['r', 'm', 'k', 'b','g','aqua']
	labels = ['Rosko', 'ARMPL','ARMCL', 'Rosko_pred']
	#
	df1 = pandas.read_csv('results_intel_test')
	sps = [80, 85, 90, 95, 98]
	layers = df1[(df1['algo'] == 'cake') & (df1['sp'] == sps[0])]['layer']._values
	cores = range(2,11)
	error = 0
	#
	for sp in sps:
		gflops_armpl=[];gflops_rosko=[];dram_io_rosko=[];dram_io_armpl=[];
		gflops_armcl=[]; dram_io_armcl=[]; dram_bw_rosko=[];dram_bw_armpl=[];
		dram_bw_armcl=[];time_armcl=[]; time_armpl=[]; time_rosko=[]; flops = []
		for p in cores:
			N = 10000
			M = 10000
			K = 10000
			nz = (1.0 - (float(sp) / 100.0))*M*K
			flops.append(nz) 
			df2 = pandas.read_csv('reports_intel_test/report_rosko_%d-%d.csv' % (p,sp) ,skiprows=17,skipfooter=16)
			cpu_time = df1[(df1['algo'] == 'rosko') & (df1['p'] == p) & (df1['sp'] == sp)]['time']._values[0]
			dram_bw = df2['Average']._values[0]
			gflops_rosko.append((nz*N) / cpu_time / 1e9)
			df2 = pandas.read_csv('reports_intel_test/report_rosko_%d-%d.csv' % (p,sp),skipfooter=20)
			elapsed = df2[df2['Metric Name'] == 'Elapsed Time']['Metric Value']._values[0]
			dram_bw_rosko.append(dram_bw)
			time_rosko.append(elapsed / ntrials)
		dram_io_rosko_pred = df1[(df1['algo'] == 'rosko_dram') & (df1['sp'] == sp) & (df1['p'].isin(cores))]['time']._values
		dram_bw_rosko_pred = dram_io_rosko_pred / np.array(time_rosko) 
		# print(dram_io_rosko_pred / np.array(dram_io_rosko))
		print(dram_bw_rosko_pred / np.array(dram_bw_rosko))
		error += np.mean(dram_bw_rosko_pred / np.array(dram_bw_rosko))
		print(dram_bw_rosko_pred)
		print(dram_bw_rosko)
		print
		#
		flops = np.log10(np.array(flops))
		plt.figure(figsize = (6,4))
		plt.scatter(flops, gflops_rosko, label = labels[0],  marker = markers[0], color = colors[0], s=20)
		# plt.scatter(flops, gflops_armpl, label = labels[1],  marker = markers[1], color = colors[1], s=20)
		# plt.scatter(flops, gflops_armcl, label = labels[2],  marker = markers[3], color = colors[2], s=20)
		#
		plt.title('(a) Throughput for SpMM in\nCNN Layers', fontsize = 24)
		plt.xlabel("# of nonzeroes (log scale)", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylabel("Throughput (GFLOPs/sec)", fontsize = 18)
		plt.legend(loc = "upper left", prop={'size': 12})
		# plt.savefig("%s_tput.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		#
		#
		plt.figure(figsize = (6,4))
		plt.scatter(flops, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		# plt.scatter(flops, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.scatter(flops, dram_bw_armcl, label = labels[2],  marker = markers[2], color = colors[2])
		plt.scatter(flops, dram_bw_rosko_pred, label = labels[3],  marker = markers[0], color = colors[3])
		#
		plt.title('(c) DRAM BW for SpMM in\nCNN Layers', fontsize = 24)
		plt.xlabel("# of nonzeros (log10 scale)", fontsize = 24)
		# plt.xticks(np.arange(3.5,5.6,0.5), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
		plt.legend(loc = "upper right", prop={'size': 12})
		# plt.savefig("%s_bw.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
		#
		#
		plt.figure(figsize = (6,4))
		plt.scatter(time_rosko, dram_bw_rosko, label = labels[0],  marker = markers[0], color = colors[0])
		# plt.scatter(time_armpl, dram_bw_armpl, label = labels[1],  marker = markers[1], color = colors[1])
		# plt.scatter(time_armcl, dram_bw_armcl, label = labels[2],  marker = markers[2], color = colors[2])
		plt.scatter(time_rosko, dram_bw_rosko_pred, label = labels[3],  marker = markers[0], color = colors[3])
		#
		plt.title('(c) BW Required to Attain\nTarget Runtime', fontsize = 24)
		plt.xlabel("Runtime (sec)", fontsize = 24)
		# plt.xticks(np.arange(0,0.31,0.05), fontsize = 16)
		plt.yticks(fontsize = 16)
		plt.ylabel("DRAM BW (GB/sec)", fontsize = 24)
		plt.legend(loc = "upper right", prop={'size': 12})
		# plt.savefig("%s_bw_tput.pdf" % fname, bbox_inches='tight')
		plt.show()
		plt.clf()
		plt.close('all')
	print(error / len(sps))





plot_rosko_vs_intel_dnn()




