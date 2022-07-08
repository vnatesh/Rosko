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


def plot_rosko_vs_intel_ablate(fname = 'rosko_vs_intel_ablate', ntrials = 2):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','k','m','r']
	labels = ['MKL', 'CAKE','Rosko', 'Rosko+reordering', 'Rosko+reorder+sp_tiling']
	sparsity = [80, 82, 85, 87, 90, 92, 95, 97, 99]
	M=K=N=10000
	dft = pandas.read_csv('result_ablate_intel')
	runtime_mkl = [dft[(dft['algo'] == 'mkl') & (dft['sp'] == 1)]['time'].mean()]*len(sparsity)
	runtime_cake = [dft[(dft['algo'] == 'cake') & (dft['sp'] == 1)]['time'].mean()]*len(sparsity)
	#
	df1 = pandas.read_csv('reports_intel_ablate/report_mkl.csv' ,skiprows=17,skipfooter=16)
	df2 = pandas.read_csv('reports_intel_ablate/report_mkl.csv' ,skipfooter=20)
	x = (df1['Average']._values[0])
	dram_bw_mkl = [x]*len(sparsity)
	dram_io_mkl = [x*runtime_mkl[0]]*len(sparsity)
	#
	df1 = pandas.read_csv('reports_intel_ablate/report_cake.csv' ,skiprows=17,skipfooter=16)
	df2 = pandas.read_csv('reports_intel_ablate/report_cake.csv' ,skipfooter=20)
	x = (df1['Average']._values[0])
	dram_bw_cake = [x]*len(sparsity)
	dram_io_cake = [x*runtime_cake[0]]*len(sparsity)
	#
	df1 = pandas.read_csv('reports_intel_ablate/report_mkl_spec.csv')
	spec_mkl = [float(df1[(df1['Metric Name'] == 'Bad Speculation')]['Metric Value']._values[0])]*len(sparsity)
	#
	df1 = pandas.read_csv('reports_intel_ablate/report_cake_spec.csv' )
	spec_cake = [float(df1[(df1['Metric Name'] == 'Bad Speculation')]['Metric Value']._values[0])]*len(sparsity)
	#
	dram_bw_rosko_reorder=[]; dram_bw_rosko=[]; runtime_rosko = []; runtime_rosko_reorder	= []
	spec_rosko = []; spec_rosko_reorder = []; dram_io_rosko_reorder = []; dram_io_rosko = [];
	spec_rosko_reorder_tile = []; dram_bw_rosko_reorder_tile =[]; runtime_rosko_reorder_tile = []; dram_io_rosko_reorder_tile = [];
	#	
	for i in range(len(sparsity)):
		cpu_time = dft[(dft['algo'] == 'rosko') & (dft['sp'] == sparsity[i])]['time'].mean()
		df1 = pandas.read_csv('reports_intel_ablate/report_rosko_%d.csv' % (sparsity[i]) ,skiprows=17,skipfooter=16)
		x = (df1['Average']._values[0])
		dram_bw_rosko.append(x)
		dram_io_rosko.append(x*cpu_time)
		runtime_rosko.append(cpu_time)
		#
		cpu_time = dft[(dft['algo'] == 'rosko+reorder') & (dft['sp'] == sparsity[i])]['time'].mean()
		df1 = pandas.read_csv('reports_intel_ablate/report_rosko_reorder_%d.csv' % (sparsity[i]) ,skiprows=17,skipfooter=16)
		x = (df1['Average']._values[0])
		dram_bw_rosko_reorder.append(x)
		dram_io_rosko_reorder.append(x*cpu_time)
		runtime_rosko_reorder.append(cpu_time)
		#
		cpu_time = dft[(dft['algo'] == 'rosko+reorder+sp_tiling') & (dft['sp'] == sparsity[i])]['time'].mean()
		df1 = pandas.read_csv('reports_intel_ablate/report_rosko_reorder_tile_%d.csv' % (sparsity[i]) ,skiprows=17,skipfooter=16)
		x = (df1['Average']._values[0])
		dram_bw_rosko_reorder_tile.append(x)
		dram_io_rosko_reorder_tile.append(x*cpu_time)
		runtime_rosko_reorder_tile.append(cpu_time)
		# df1 = pandas.read_csv('reports_intel_ablate/report_rosko_spec_%d-%d.csv' % (sparsity[i], ))
		# spec_rosko.append(float(df1[(df1['Metric Name'] == 'Bad Speculation')]['Metric Value']._values[0]))
		# #
		# df1 = pandas.read_csv('reports_intel_ablate/report_rosko_reorder_spec_%d-%d.csv' % (sparsity[i]))
		# spec_rosko_reorder.append(float(df1[(df1['Metric Name'] == 'Bad Speculation')]['Metric Value']._values[0]))
	#
	#
	for i in range(len(sparsity)):
		df1 = pandas.read_csv('reports_intel_ablate/report_rosko_spec_%d.csv' % sparsity[i])
		spec_rosko.append(float(df1[(df1['Metric Name'] == 'Bad Speculation')]['Metric Value']._values[0]))
		#
		df1 = pandas.read_csv('reports_intel_ablate/report_rosko_reorder_spec_%d.csv' % sparsity[i])
		spec_rosko_reorder.append(float(df1[(df1['Metric Name'] == 'Bad Speculation')]['Metric Value']._values[0]))
		#
		df1 = pandas.read_csv('reports_intel_ablate/report_rosko_reorder_tile_spec_%d.csv' % sparsity[i])
		spec_rosko_reorder_tile.append(float(df1[(df1['Metric Name'] == 'Bad Speculation')]['Metric Value']._values[0]))
		#
	#
	# plt.subplot(1, 2, 1)
	plt.figure(figsize = (6,4))
	plt.plot(sparsity, dram_bw_mkl, label = labels[0], color = colors[0])
	plt.plot(sparsity, dram_bw_cake, label = labels[1], color = colors[1])
	plt.plot(sparsity, dram_bw_rosko, label = labels[2], color = colors[2])
	plt.plot(sparsity, dram_bw_rosko_reorder, label = labels[3], color = colors[3])
	plt.plot(sparsity, dram_bw_rosko_reorder_tile, label = labels[4], color = colors[4])
	#
	plt.title('(e) DRAM Bandwidth on Intel-i9', fontsize = 24)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.xticks(range(80,101,5), fontsize = 20)
	plt.yticks(range(0,19,3), fontsize = 16)
	plt.ylabel("DRAM Bw (GB/s)", fontsize = 24)
	# plt.legend(loc = "upper left", prop={'size': 14})
	plt.savefig("%s_dram.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	tmlk = [runtime_mkl[i]/runtime_mkl[i] for i in range(len(sparsity))]
	tcake = [runtime_mkl[i]/runtime_cake[i] for i in range(len(sparsity))]
	trosko = [runtime_mkl[i]/runtime_rosko[i] for i in range(len(sparsity))]
	troskoreorder = [runtime_mkl[i]/runtime_rosko_reorder[i] for i in range(len(sparsity))]
	troskoreordertile = [runtime_mkl[i]/runtime_rosko_reorder_tile[i] for i in range(len(sparsity))]
	plt.figure(figsize = (6,4))
	plt.plot(sparsity, tmlk, label = labels[0], color = colors[0])
	plt.plot(sparsity, tcake, label = labels[1], color = colors[1])
	plt.plot(sparsity, trosko, label = labels[2], color = colors[2])
	plt.plot(sparsity, troskoreorder, label = labels[3],  color = colors[3])
	plt.plot(sparsity, troskoreordertile, label = labels[4],  color = colors[4])
	#
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(b) Throughput on Intel-i9', fontsize = 24)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.ylabel("Relative Tput wrt MKL", fontsize = 24)
	plt.xticks(range(80,101,5), fontsize = 20)
	plt.yticks(np.arange(0,7,1), fontsize = 20)
	plt.legend(loc = "upper left", prop={'size': 16})
	plt.savefig("%s_perf.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(sparsity, spec_mkl, label = labels[0], color = colors[0])
	plt.plot(sparsity, spec_cake, label = labels[1], color = colors[1])
	plt.plot(sparsity, spec_rosko, label = labels[2], color = colors[2])
	plt.plot(sparsity, spec_rosko_reorder, label = labels[3],  color = colors[3])
	plt.plot(sparsity, spec_rosko_reorder_tile, label = labels[4],  color = colors[4])
	#
	plt.ticklabel_format(useOffset=False, style='plain')
	plt.title('(f) Misspeculation on Intel-i9', fontsize = 24)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.ylabel("Pipeline Slots (%)", fontsize = 24)
	plt.xticks(range(80,101,5), fontsize = 20)
	plt.yticks(range(0,19,2), fontsize = 20)
	# plt.legend(loc = "center left", prop={'size': 14})
	plt.savefig("%s_spec.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')
	#
	plt.figure(figsize = (6,4))
	plt.plot(sparsity, dram_io_mkl, label = labels[0], color = colors[0])
	plt.plot(sparsity, dram_io_cake, label = labels[1], color = colors[1])
	plt.plot(sparsity, dram_io_rosko, label = labels[2], color = colors[2])
	plt.plot(sparsity, dram_io_rosko_reorder, label = labels[3], color = colors[3])
	plt.plot(sparsity, dram_io_rosko_reorder_tile, label = labels[4], color = colors[4])
	#
	plt.title('(e) DRAM IO on Intel-i9', fontsize = 24)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.ylabel("DRAM IO (GB)", fontsize = 24)
	plt.xticks(range(80,101,5), fontsize = 20)
	plt.yticks(fontsize = 20)
	# plt.legend(loc = "center right", prop={'size': 14})
	plt.savefig("%s_io.pdf" % fname, bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')


plot_rosko_vs_intel_ablate()

