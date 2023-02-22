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

rosko_vs_mkl_suitesparse()




df1[df1['file'].isin(mkl_file)]

import ssgetpy
import os
import shutil

group = '', 
files = ssgetpy.search(rowbounds=(16000,25000),colbounds=(16000,25000), \
    nzbounds = (1000000, None), dtype = 'real', limit = 300)

files = ssgetpy.search(rowbounds=(1000,25000),colbounds=(1000,25000), \
     group = 'ML_Graph', dtype = 'real', limit = 300)


files = ssgetpy.search(rowbounds=(1000,5000),colbounds=(1000,5000), \
     nzbounds = (100000, None), dtype = 'real', limit = 300)


files = ssgetpy.search(rowbounds=(1000,25000),colbounds=(1000,25000), \
  		 dtype = 'real', limit = 3000)

new =[]
for i in files:
	if (i.nsym == 0.0) and ((1 - (i.nnz / (i.rows*i.cols))) <= 0.995):
		new.append(i)



def rosko_vs_mkl_suitesparse(fname = 'rosko_vs_mkl_suite_new'):
	plt.rcParams.update({'font.size': 12})
	denser = ["data/airfoil_2d.mtx","data/appu.mtx","data/bas1lp.mtx","data/bcsstk17.mtx","data/bcsstk25.mtx","data/bcsstk38.mtx","data/benzene.mtx","data/bundle1.mtx","data/cbuckle.mtx","data/crystk02.mtx","data/crystm02.mtx","data/cyl6.mtx","data/ex19.mtx","data/ex40.mtx","data/exdata_1.mtx","data/fp.mtx","data/garon2.mtx","data/goodwin.mtx","data/graham1.mtx","data/human_gene2.mtx","data/IG5-15.mtx","data/inlet.mtx","data/k3plates.mtx","data/Kuu.mtx","data/lhr14c.mtx","data/lhr14.mtx","data/msc10848.mtx","data/Na5.mtx","data/nd3k.mtx","data/nemeth01.mtx","data/nemeth02.mtx","data/nemeth03.mtx","data/nemeth04.mtx","data/nemeth05.mtx","data/nemeth06.mtx","data/nemeth07.mtx","data/nemeth08.mtx","data/nemeth09.mtx","data/nemeth10.mtx","data/nemeth11.mtx","data/nemeth12.mtx","data/nemeth13.mtx","data/nemeth14.mtx","data/nemeth15.mtx","data/nemeth16.mtx","data/nemeth17.mtx","data/nemeth18.mtx","data/nemeth19.mtx","data/nemeth20.mtx","data/nemeth21.mtx","data/nemeth22.mtx","data/nemeth23.mtx","data/nemeth24.mtx","data/nemeth25.mtx","data/nemeth26.mtx","data/poisson3Da.mtx","data/Pres_Poisson.mtx","data/Reuters911.mtx","data/s1rmq4m1.mtx","data/s2rmq4m1.mtx","data/s3rmq4m1.mtx","data/sinc12.mtx","data/sinc15.mtx","data/sme3Da.mtx","data/t520.mtx","data/ted_A.mtx","data/ted_A_unscaled.mtx","data/TSOPF_FS_b162_c1.mtx","data/TSOPF_RS_b162_c3.mtx","data/TSOPF_RS_b300_c1.mtx","data/TSOPF_RS_b39_c7.mtx","data/vibrobox.mtx"]
	nonsym = ["data/airfoil_2d.mtx","data/appu.mtx","data/bas1lp.mtx","data/bayer02.mtx","data/bayer03.mtx","data/bayer10.mtx","data/big.mtx","data/cage10.mtx","data/cell1.mtx","data/cell2.mtx","data/ch7-6-b4.mtx","data/ch7-6-b5.mtx","data/circuit_3.mtx","data/co5.mtx","data/coater2.mtx","data/coupled.mtx","data/cq5.mtx","data/cryg10000.mtx","data/deter1.mtx","data/deter5.mtx","data/dw4096.mtx","data/dw8192.mtx","data/epb1.mtx","data/ex18.mtx","data/ex19.mtx","data/ex40.mtx","data/FA.mtx","data/fd12.mtx","data/fd15.mtx","data/flowmeter5.mtx","data/foldoc.mtx","data/fp.mtx","data/Franz4.mtx","data/fxm3_6.mtx","data/g7jac020.mtx","data/g7jac020sc.mtx","data/g7jac040.mtx","data/g7jac040sc.mtx","data/g7jac050sc.mtx","data/garon2.mtx","data/goodwin.mtx","data/graham1.mtx","data/Hamrle2.mtx","data/hydr1c.mtx","data/hydr1.mtx","data/IG5-14.mtx","data/IG5-15.mtx","data/igbt3.mtx","data/inlet.mtx","data/jan99jac020.mtx","data/jan99jac020sc.mtx","data/jan99jac040.mtx","data/jan99jac040sc.mtx","data/k3plates.mtx","data/Kaufhold.mtx","data/Lederberg.mtx","data/lhr07c.mtx","data/lhr07.mtx","data/lhr10c.mtx","data/lhr10.mtx","data/lhr11c.mtx","data/lhr11.mtx","data/lhr14c.mtx","data/lhr14.mtx","data/lp_dfl001.mtx","data/mark3jac020.mtx","data/mark3jac020sc.mtx","data/n3c6-b6.mtx","data/n3c6-b7.mtx","data/nl.mtx","data/olm5000.mtx","data/p05.mtx","data/Pd.mtx","data/Pd_rhs.mtx","data/pesa.mtx","data/poisson3Da.mtx","data/poli_large.mtx","data/powersim.mtx","data/psse1.mtx","data/r05.mtx","data/raefsky5.mtx","data/rajat03.mtx","data/rajat13.mtx","data/rdb5000.mtx","data/rw5151.mtx","data/scagr7-2b.mtx","data/sherman3.mtx","data/shermanACd.mtx","data/sinc12.mtx","data/sinc15.mtx","data/sme3Da.mtx","data/t2d_q4.mtx","data/t2d_q9.mtx","data/ted_A.mtx","data/ted_A_unscaled.mtx","data/TF15.mtx","data/TSOPF_RS_b162_c1.mtx","data/TSOPF_RS_b162_c3.mtx","data/TSOPF_RS_b300_c1.mtx","data/TSOPF_RS_b39_c7.mtx","data/TSOPF_RS_b9_c6.mtx","data/utm5940.mtx","data/Zewail.mtx"]
	syms = ["data/aft01.mtx","data/Alemdar.mtx","data/bcsstk17.mtx","data/bcsstk18.mtx","data/bcsstk25.mtx","data/bcsstk38.mtx","data/bcsstm25.mtx","data/bcsstm38.mtx","data/benzene.mtx","data/bloweybq.mtx","data/bundle1.mtx","data/c-29.mtx","data/c-30.mtx","data/c-31.mtx","data/c-32.mtx","data/c-33.mtx","data/c-34.mtx","data/c-35.mtx","data/c-36.mtx","data/c-37.mtx","data/c-38.mtx","data/c-39.mtx","data/c-40.mtx","data/c-41.mtx","data/c-42.mtx","data/c-43.mtx","data/c-44.mtx","data/c-45.mtx","data/c-46.mtx","data/c-47.mtx","data/case9.mtx","data/cbuckle.mtx","data/crystk02.mtx","data/crystm02.mtx","data/cyl6.mtx","data/eurqsa.mtx","data/ex15.mtx","data/exdata_1.mtx","data/flowmeter0.mtx","data/fv1.mtx","data/fv2.mtx","data/fv3.mtx","data/G56.mtx","data/G57.mtx","data/G59.mtx","data/G61.mtx","data/G62.mtx","data/G64.mtx","data/G65.mtx","data/G66.mtx","data/G67.mtx","data/geom.mtx","data/human_gene2.mtx","data/Kuu.mtx","data/linverse.mtx","data/m3plates.mtx","data/meg4.mtx","data/msc10848.mtx","data/Muu.mtx","data/Na5.mtx","data/ncvxqp1.mtx","data/nd3k.mtx","data/nemeth01.mtx","data/nemeth02.mtx","data/nemeth03.mtx","data/nemeth04.mtx","data/nemeth05.mtx","data/nemeth06.mtx","data/nemeth07.mtx","data/nemeth08.mtx","data/nemeth09.mtx","data/nemeth10.mtx","data/nemeth11.mtx","data/nemeth12.mtx","data/nemeth13.mtx","data/nemeth14.mtx","data/nemeth15.mtx","data/nemeth16.mtx","data/nemeth17.mtx","data/nemeth18.mtx","data/nemeth19.mtx","data/nemeth20.mtx","data/nemeth21.mtx","data/nemeth22.mtx","data/nemeth23.mtx","data/nemeth24.mtx","data/nemeth25.mtx","data/nemeth26.mtx","data/nopoly.mtx","data/Pres_Poisson.mtx","data/rail_5177.mtx","data/Reuters911.mtx","data/s1rmq4m1.mtx","data/s1rmt3m1.mtx","data/s2rmq4m1.mtx","data/s2rmt3m1.mtx","data/s3rmq4m1.mtx","data/s3rmt3m1.mtx","data/s3rmt3m3.mtx","data/SiH4.mtx","data/SiNa.mtx","data/sit100.mtx","data/stokes64.mtx","data/stokes64s.mtx","data/t2dah_a.mtx","data/t2dah_e.mtx","data/t2dah.mtx","data/t520.mtx","data/ted_B.mtx","data/ted_B_unscaled.mtx","data/TSOPF_FS_b162_c1.mtx","data/TSOPF_FS_b9_c6.mtx","data/tuma2.mtx","data/vibrobox.mtx"]
	df1 = pandas.read_csv('result_sp_new')
	df1 = df1[df1['file'].isin(denser)]
	labels = ['TUMMY Wins','MKL Wins']
	rel_tput = df1[df1['algo'] == 'mkl']['time']._values / df1[df1['algo'] == 'rosko']['time']._values
	sparsity = df1[df1['algo'] == 'mkl']['sparsity']._values
	files = df1[df1['algo']=='rosko']['file'].values
	mkl_file = []
	rosko_file = []
	rel_tput1=[]
	sparsity1=[]
	sp_lim = 90
	for i in xrange(len(rel_tput)):
		if sparsity[i] <= 100 and sparsity[i] >= sp_lim:
			sparsity1.append(sparsity[i])
			if rel_tput[i] < 1.0:
				rel_tput1.append(-1.0/rel_tput[i])
				mkl_file.append(files[i])
			else:
				rel_tput1.append(rel_tput[i])
				rosko_file.append(files[i])
	# sparsity1 = [i*100 for i in sparsity1]
	sp_rosko=[]; sp_mkl=[]; rosko_tput=[]; mkl_tput=[]
	for i in range(len(rel_tput1)):
		if rel_tput1[i] > 0:
			rosko_tput.append(rel_tput1[i])
			sp_rosko.append(sparsity1[i])
		else:
			mkl_tput.append(rel_tput1[i])
			sp_mkl.append(sparsity1[i])
	plt.figure(figsize = (6,5))
	plt.title('(a) TUMMY vs MKL SpMM Throughput\nfor SuiteSparse Matrices', fontsize = 24)
	plt.scatter(sp_rosko, rosko_tput, color = 'r', label = labels[0])
	plt.scatter(sp_mkl, mkl_tput, color = intel_color, label = labels[1])
	plt.legend(loc = "lower left", labels=labels, prop={'size': 16})
	plt.plot([0,100], [1,1], color = 'k')
	plt.plot([0,100], [-1,-1], color = 'k')
	plt.xticks(np.arange(sp_lim,101,0.5), fontsize = 14)
	plt.yticks([-3, -2, -1, 1, 2, 4, 6, 8], fontsize = 14)
	plt.xlim(sp_lim, 100.1)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.ylabel("TUMMY Tput Relative to MKL", fontsize = 18)
	plt.tight_layout()
	# plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
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
	plt.title('(b) TUMMY vs MKL SpMM DRAM BW\nfor SuiteSparse Matrices', fontsize = 24)
	plt.scatter(sparsity1, rosko_bw, color = 'r', label = labels[0])
	plt.scatter(sparsity1, mkl_bw, color = intel_color, label = labels[1])
	plt.legend(loc = "upper right", labels=labels, prop={'size': 16})
	plt.xticks(np.arange(98,101,0.5), fontsize = 14)
	plt.xlim(98, 100)
	plt.ylim(0, 18)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.ylabel("DRAM Bandwidth (GB/sec)", fontsize = 18)
	plt.tight_layout()
	plt.savefig("%s_dram.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')




rosko_vs_mkl_suitesparse()











def rosko_vs_mkl_suitesparse(fname = 'rosko_vs_mkl_suite_mlgraph'):
	plt.rcParams.update({'font.size': 12})
	df1 = pandas.read_csv('result_sp_new2')
	labels = ['TUMMY Wins','MKL Wins']
	rel_tput = df1[df1['algo'] == 'mkl']['time']._values / df1[df1['algo'] == 'rosko']['time']._values
	sparsity = df1[df1['algo'] == 'mkl']['sparsity']._values
	files = df1[df1['algo']=='rosko']['file'].values
	mkl_file = []
	rosko_file = []
	rel_tput1=[]
	sparsity1=[]
	sp_lim = 90
	for i in xrange(len(rel_tput)):
		if sparsity[i] <= 100 and sparsity[i] >= sp_lim:
			sparsity1.append(sparsity[i])
			if rel_tput[i] < 1.0:
				rel_tput1.append(-1.0/rel_tput[i])
				mkl_file.append(files[i])
			else:
				rel_tput1.append(rel_tput[i])
				rosko_file.append(files[i])
	# sparsity1 = [i*100 for i in sparsity1]
	sp_rosko=[]; sp_mkl=[]; rosko_tput=[]; mkl_tput=[]
	for i in range(len(rel_tput1)):
		if rel_tput1[i] > 0:
			rosko_tput.append(rel_tput1[i])
			sp_rosko.append(sparsity1[i])
		else:
			mkl_tput.append(rel_tput1[i])
			sp_mkl.append(sparsity1[i])
	plt.figure(figsize = (6,5))
	plt.title('(a) TUMMY vs MKL SpMM Throughput\nfor SuiteSparse Matrices', fontsize = 24)
	plt.scatter(sp_rosko, rosko_tput, color = 'r', label = labels[0])
	plt.scatter(sp_mkl, mkl_tput, color = intel_color, label = labels[1])
	plt.legend(loc = "lower left", labels=labels, prop={'size': 16})
	plt.plot([0,100], [1,1], color = 'k')
	plt.plot([0,100], [-1,-1], color = 'k')
	plt.xticks(np.arange(sp_lim,101,0.5), fontsize = 14)
	plt.yticks([-3, -2, -1, 1, 2, 4, 6, 8], fontsize = 14)
	plt.xlim(sp_lim, 100.1)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.ylabel("TUMMY Tput Relative to MKL", fontsize = 18)
	plt.tight_layout()
	# plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')




















def rosko_vs_mkl_suitesparse(fname = 'rosko_vs_mkl_suite_new'):
	plt.rcParams.update({'font.size': 12})
	# denser = ["data/airfoil_2d.mtx","data/appu.mtx","data/bas1lp.mtx","data/bcsstk17.mtx","data/bcsstk25.mtx","data/bcsstk38.mtx","data/benzene.mtx","data/bundle1.mtx","data/cbuckle.mtx","data/crystk02.mtx","data/crystm02.mtx","data/cyl6.mtx","data/ex19.mtx","data/ex40.mtx","data/exdata_1.mtx","data/fp.mtx","data/garon2.mtx","data/goodwin.mtx","data/graham1.mtx","data/human_gene2.mtx","data/IG5-15.mtx","data/inlet.mtx","data/k3plates.mtx","data/Kuu.mtx","data/lhr14c.mtx","data/lhr14.mtx","data/msc10848.mtx","data/Na5.mtx","data/nd3k.mtx","data/nemeth01.mtx","data/nemeth02.mtx","data/nemeth03.mtx","data/nemeth04.mtx","data/nemeth05.mtx","data/nemeth06.mtx","data/nemeth07.mtx","data/nemeth08.mtx","data/nemeth09.mtx","data/nemeth10.mtx","data/nemeth11.mtx","data/nemeth12.mtx","data/nemeth13.mtx","data/nemeth14.mtx","data/nemeth15.mtx","data/nemeth16.mtx","data/nemeth17.mtx","data/nemeth18.mtx","data/nemeth19.mtx","data/nemeth20.mtx","data/nemeth21.mtx","data/nemeth22.mtx","data/nemeth23.mtx","data/nemeth24.mtx","data/nemeth25.mtx","data/nemeth26.mtx","data/poisson3Da.mtx","data/Pres_Poisson.mtx","data/Reuters911.mtx","data/s1rmq4m1.mtx","data/s2rmq4m1.mtx","data/s3rmq4m1.mtx","data/sinc12.mtx","data/sinc15.mtx","data/sme3Da.mtx","data/t520.mtx","data/ted_A.mtx","data/ted_A_unscaled.mtx","data/TSOPF_FS_b162_c1.mtx","data/TSOPF_RS_b162_c3.mtx","data/TSOPF_RS_b300_c1.mtx","data/TSOPF_RS_b39_c7.mtx","data/vibrobox.mtx"]
	# nonsym = ["data/airfoil_2d.mtx","data/appu.mtx","data/bas1lp.mtx","data/bayer02.mtx","data/bayer03.mtx","data/bayer10.mtx","data/big.mtx","data/cage10.mtx","data/cell1.mtx","data/cell2.mtx","data/ch7-6-b4.mtx","data/ch7-6-b5.mtx","data/circuit_3.mtx","data/co5.mtx","data/coater2.mtx","data/coupled.mtx","data/cq5.mtx","data/cryg10000.mtx","data/deter1.mtx","data/deter5.mtx","data/dw4096.mtx","data/dw8192.mtx","data/epb1.mtx","data/ex18.mtx","data/ex19.mtx","data/ex40.mtx","data/FA.mtx","data/fd12.mtx","data/fd15.mtx","data/flowmeter5.mtx","data/foldoc.mtx","data/fp.mtx","data/Franz4.mtx","data/fxm3_6.mtx","data/g7jac020.mtx","data/g7jac020sc.mtx","data/g7jac040.mtx","data/g7jac040sc.mtx","data/g7jac050sc.mtx","data/garon2.mtx","data/goodwin.mtx","data/graham1.mtx","data/Hamrle2.mtx","data/hydr1c.mtx","data/hydr1.mtx","data/IG5-14.mtx","data/IG5-15.mtx","data/igbt3.mtx","data/inlet.mtx","data/jan99jac020.mtx","data/jan99jac020sc.mtx","data/jan99jac040.mtx","data/jan99jac040sc.mtx","data/k3plates.mtx","data/Kaufhold.mtx","data/Lederberg.mtx","data/lhr07c.mtx","data/lhr07.mtx","data/lhr10c.mtx","data/lhr10.mtx","data/lhr11c.mtx","data/lhr11.mtx","data/lhr14c.mtx","data/lhr14.mtx","data/lp_dfl001.mtx","data/mark3jac020.mtx","data/mark3jac020sc.mtx","data/n3c6-b6.mtx","data/n3c6-b7.mtx","data/nl.mtx","data/olm5000.mtx","data/p05.mtx","data/Pd.mtx","data/Pd_rhs.mtx","data/pesa.mtx","data/poisson3Da.mtx","data/poli_large.mtx","data/powersim.mtx","data/psse1.mtx","data/r05.mtx","data/raefsky5.mtx","data/rajat03.mtx","data/rajat13.mtx","data/rdb5000.mtx","data/rw5151.mtx","data/scagr7-2b.mtx","data/sherman3.mtx","data/shermanACd.mtx","data/sinc12.mtx","data/sinc15.mtx","data/sme3Da.mtx","data/t2d_q4.mtx","data/t2d_q9.mtx","data/ted_A.mtx","data/ted_A_unscaled.mtx","data/TF15.mtx","data/TSOPF_RS_b162_c1.mtx","data/TSOPF_RS_b162_c3.mtx","data/TSOPF_RS_b300_c1.mtx","data/TSOPF_RS_b39_c7.mtx","data/TSOPF_RS_b9_c6.mtx","data/utm5940.mtx","data/Zewail.mtx"]
	# syms = ["data/aft01.mtx","data/Alemdar.mtx","data/bcsstk17.mtx","data/bcsstk18.mtx","data/bcsstk25.mtx","data/bcsstk38.mtx","data/bcsstm25.mtx","data/bcsstm38.mtx","data/benzene.mtx","data/bloweybq.mtx","data/bundle1.mtx","data/c-29.mtx","data/c-30.mtx","data/c-31.mtx","data/c-32.mtx","data/c-33.mtx","data/c-34.mtx","data/c-35.mtx","data/c-36.mtx","data/c-37.mtx","data/c-38.mtx","data/c-39.mtx","data/c-40.mtx","data/c-41.mtx","data/c-42.mtx","data/c-43.mtx","data/c-44.mtx","data/c-45.mtx","data/c-46.mtx","data/c-47.mtx","data/case9.mtx","data/cbuckle.mtx","data/crystk02.mtx","data/crystm02.mtx","data/cyl6.mtx","data/eurqsa.mtx","data/ex15.mtx","data/exdata_1.mtx","data/flowmeter0.mtx","data/fv1.mtx","data/fv2.mtx","data/fv3.mtx","data/G56.mtx","data/G57.mtx","data/G59.mtx","data/G61.mtx","data/G62.mtx","data/G64.mtx","data/G65.mtx","data/G66.mtx","data/G67.mtx","data/geom.mtx","data/human_gene2.mtx","data/Kuu.mtx","data/linverse.mtx","data/m3plates.mtx","data/meg4.mtx","data/msc10848.mtx","data/Muu.mtx","data/Na5.mtx","data/ncvxqp1.mtx","data/nd3k.mtx","data/nemeth01.mtx","data/nemeth02.mtx","data/nemeth03.mtx","data/nemeth04.mtx","data/nemeth05.mtx","data/nemeth06.mtx","data/nemeth07.mtx","data/nemeth08.mtx","data/nemeth09.mtx","data/nemeth10.mtx","data/nemeth11.mtx","data/nemeth12.mtx","data/nemeth13.mtx","data/nemeth14.mtx","data/nemeth15.mtx","data/nemeth16.mtx","data/nemeth17.mtx","data/nemeth18.mtx","data/nemeth19.mtx","data/nemeth20.mtx","data/nemeth21.mtx","data/nemeth22.mtx","data/nemeth23.mtx","data/nemeth24.mtx","data/nemeth25.mtx","data/nemeth26.mtx","data/nopoly.mtx","data/Pres_Poisson.mtx","data/rail_5177.mtx","data/Reuters911.mtx","data/s1rmq4m1.mtx","data/s1rmt3m1.mtx","data/s2rmq4m1.mtx","data/s2rmt3m1.mtx","data/s3rmq4m1.mtx","data/s3rmt3m1.mtx","data/s3rmt3m3.mtx","data/SiH4.mtx","data/SiNa.mtx","data/sit100.mtx","data/stokes64.mtx","data/stokes64s.mtx","data/t2dah_a.mtx","data/t2dah_e.mtx","data/t2dah.mtx","data/t520.mtx","data/ted_B.mtx","data/ted_B_unscaled.mtx","data/TSOPF_FS_b162_c1.mtx","data/TSOPF_FS_b9_c6.mtx","data/tuma2.mtx","data/vibrobox.mtx"]
	df1 = pandas.read_csv('result_sp_99')
	# df1 = df1[df1['file'].isin(denser)]
	diffs = set(df1[df1['algo'] == 'mkl']['file']) - set(df1[df1['algo'] == 'rosko']['file'])
	df1 = df1[~df1['file'].isin(diffs)]
	labels = ['Rosko Wins','MKL Wins']
	rel_tput = df1[df1['algo'] == 'mkl']['time']._values / df1[df1['algo'] == 'rosko']['time']._values
	sparsity = df1[df1['algo'] == 'mkl']['sparsity']._values
	files = df1[df1['algo']=='rosko']['file'].values
	mkl_file = []
	rosko_file = []
	rel_tput1=[]
	sparsity1=[]
	sp_min = 90
	sp_max = 99.9999
	for i in xrange(len(rel_tput)):
		if sparsity[i] <= 100 and (sparsity[i] >= sp_min and sparsity[i] <= sp_max):
			sparsity1.append(sparsity[i])
			if rel_tput[i] < 0.95:
				rel_tput1.append(-1.0/rel_tput[i])
				mkl_file.append(files[i])
			else:
				rel_tput1.append(rel_tput[i])
				rosko_file.append(files[i])
	# sparsity1 = [i*100 for i in sparsity1]
	sp_rosko=[]; sp_mkl=[]; rosko_tput=[]; mkl_tput=[]
	for i in range(len(rel_tput1)):
		if rel_tput1[i] > 0:
			rosko_tput.append(rel_tput1[i])
			sp_rosko.append(sparsity1[i])
		else:
			mkl_tput.append(rel_tput1[i])
			sp_mkl.append(sparsity1[i])
	plt.figure(figsize = (6,5))
	plt.title('(a) TUMMY vs MKL SpMM Throughput\nfor SuiteSparse Matrices', fontsize = 24)
	plt.scatter(sp_rosko, rosko_tput, color = 'r', label = labels[0])
	plt.scatter(sp_mkl, mkl_tput, color = intel_color, label = labels[1])
	plt.legend(loc = "lower left", labels=labels, prop={'size': 16})
	plt.plot([0,100], [1,1], color = 'k')
	plt.plot([0,100], [-1,-1], color = 'k')
	plt.xticks(np.arange(sp_lim,101,0.5), fontsize = 14)
	plt.yticks([-3, -2, -1, 1, 2, 4, 6, 8], fontsize = 14)
	plt.xlim(sp_lim, 100.1)
	plt.xlabel("Sparsity (%)", fontsize = 24)
	plt.ylabel("TUMMY Tput Relative to MKL", fontsize = 18)
	plt.tight_layout()
	# plt.savefig("%s_perf.pdf" % fname , bbox_inches='tight')
	plt.show()
	plt.clf()
	plt.close('all')



rosko_vs_mkl_suitesparse()