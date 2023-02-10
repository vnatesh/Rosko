import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import os
import re
import sys



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def rosko_heatmap(fname = 'rosko_heatmap'):
	plt.rcParams.update({'font.size': 12})
	markers = ['o','v','s','d','^']
	colors = ['b','g','aqua','k','m','r','c']
	df1 = pandas.read_csv('results_mr_nr1')
	df1 = df1[(df1['nr'] != 16)]
	labels = ['rosko new','rosko old', 'cake']
	sparsity = sorted(set(df1[df1['algo'] == 'rosko_new']['sp']._values))
	mrs = df1[(df1['algo'] == 'rosko_new') & (df1['sp'] == 90)]['mr']._values
	nrs = df1[(df1['algo'] == 'rosko_new') & (df1['sp'] == 90)]['nr']._values
	num_mrs = len(set(mrs))
	num_nrs = len(set(nrs))
		# tput = np.transpose(tput)
	# heatmap, _, _ = np.histogram2d(mrs, nrs, weights=times)
	plt.figure(figsize = (8,6))
	w= []
	# sparsity = [80,90,98,99,99.5]
	sparsity = [79,87,95,98,99,99.5]
	# sparsity = [79,87,95,99]
	for i in range(len(sparsity)):
		# ax = plt.subplot(4,2, i+1)
		ax = plt.subplot(2,3, i+1)
		tput = (10000**3) / df1[(df1['algo'] == 'rosko_new') & (df1['sp'] == sparsity[i])]['time']._values
		tput = tput.reshape(num_mrs, num_nrs)
		res = np.empty((num_mrs, num_nrs))
		q = tput.reshape(num_mrs*num_nrs)
		mean = np.mean(q)
		std = np.std(q)
		mins = min(tput.reshape(num_mrs*num_nrs))
		maxs = max(tput.reshape(num_mrs*num_nrs))
		orig_map=plt.cm.get_cmap('hot')
		reversed_map = orig_map.reversed()
		for j in range(len(tput)):
			# res[j] = tput[j] / max(tput.reshape(num_mrs*num_nrs))
			# res[j] = (tput[j] - mean) / std
			# res[j] = (tput[j] - mins) / (maxs - mins)
			res[j] = tput[j]
		# ax.imshow(tput, interpolation='nearest', cmap='hot', extent=[32,96,8,20], origin='lower', aspect='auto')
		vmin = min(res.reshape(num_mrs*num_nrs))
		vmax = max(res.reshape(num_mrs*num_nrs))
		ax.imshow(res, interpolation='nearest', cmap=reversed_map, extent=[32,96,6,20], origin='lower', aspect='auto', vmin = vmin, vmax = vmax)
		if i == 0 or i == 3:
			plt.ylabel("mr", fontsize = 16, fontweight="bold")
			plt.yticks(range(6,21,2), fontsize = 13)
		else:
			plt.tick_params(left = False, right = False , labelleft = False)
		# if i > 2:
		plt.xlabel("nr", fontsize = 16, fontweight="bold")
		plt.xticks(range(32,97,16), fontsize = 14)
		plt.title('Sparsity = %.2f' % sparsity[i], fontsize = 14, fontweight="bold")
		forceAspect(ax,aspect=1)
	plt.savefig("%s.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



rosko_heatmap()



