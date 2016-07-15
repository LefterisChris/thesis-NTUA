#!/home/lefteris/anaconda2/bin/python2.7

import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def get_rating(infile,Mflops):
	cpi = np.zeros(2880)
	power = np.zeros(2880)
	ratings = np.zeros(2880)
	bw = np.zeros(2880)
	flops = np.zeros(2880)
	threads = np.zeros(2880)
	cores = np.zeros(2880)
	rtime = np.zeros(2880)
	conf = []
	i=0
	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		for row in myreader:
			""" 
			Configuration,grade1,CPI,BW/s((GB/sec),GFLOPS/s,Vectorization,Power(W),Temperature(C),Time(sec)
			initial test: take CPI 
			"""
			conf.append(row[0][1:-1].split(","))
			threads[i] = int(conf[i][7][2:-1])
			cores[i] = int(conf[i][6][2:-1])
			cpi[i] = float(row[2])
			power[i] = float(row[6])
			bw[i] = float(row[3])
			flops[i] = float(row[4])
			rtime[i] = sum(map(float,row[8].strip('[]').split(','))) / 4.0
			ratings[i] = 1000.0*(threads[i] / cpi[i]) / power[i]
			# ratings[i] = 1000.0*(cpi[i] / threads[i]) / power[i]
			# ratings[i] = (Mflops/rtime[i])
			# ratings[i] = 10.0*(Mflops/rtime[i])/power[i]
			# ratings[i] = Mflops/rtime[i]
			i += 1
	return ratings

def normalizer(infile, Mflops):
	with open(infile,'rb') as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		line = myreader.next()
		power = float(line[6])
		rtime = sum(map(float,line[8].strip('[]').split(','))) / 4.0
	return (10.0*(Mflops/rtime)/power)
	# return (Mflops/rtime)

if __name__ == '__main__':

	inRodinia = [("rodinia/nn/nn_filelist1G_8.csv",182.4), ("rodinia/kmeans/kmeans.csv",63492.0), 
				("rodinia/backprop/backprop_4194304.csv",469.8), ("rodinia/cfd/cfd_097K.csv", 157347.4),
				("rodinia/hotspot3D/hotspot3D_512_4.csv",3770.0), ("rodinia/sradv1/srad_v1.csv", 103462.0),
				("rodinia/streamcluster/streamcluster.csv",1716.0), ("rodinia/lud/lud_8000.csv", 350950.0),
				("rodinia/myocyte/myocyte_30_228.csv",2331.2), ("rodinia/lavaMD/lavaMD_20.csv", 14720.0),
				("rodinia/heartwall/heartwall_30.csv",175.9), ("rodinia/hotspot/hotspot_1024.csv", 3144.5),
				("rodinia/sradv2/srad_v2_6000.csv",151200.0), ("rodinia/pre_euler/pre_euler3d_097K.csv",168371.0)]
	labelsRodinia = [f.split('/')[1] for (f,_) in inRodinia]

	inNPB = [	("NPB/bt_A.csv",168300.0), ("NPB/ft_B.csv",92050.0), ("NPB/sp_A.csv",85000.0), ("NPB/cg_B.csv",54700.0), 
				("NPB/lu_A.csv",119280.0), ("NPB/mg_C.csv",155700.0)]
	labelsNPB = [f.split('/')[1].split('.')[0] for (f,_) in inNPB]

	baseRod = [	'base_runs/base_nn_filelist_1G.csv', 'base_runs/base_kmeans.csv',
				'base_runs/base_backprop_4194304.csv', 'base_runs/base_euler3d_097K.csv',
				'base_runs/base_hotspot3D_512_4.csv', 'base_runs/base_srad_v1.csv',
				'base_runs/base_streamcluster.csv', 'base_runs/base_lud_8000.csv',
				'base_runs/base_myocyte_30_228.csv', 'base_runs/base_lavaMD.csv',
				'base_runs/base_heartwall_30.csv', 'base_runs/base_hotspot_1024.csv',
				'base_runs/base_srad_v2_6000.csv', 'base_runs/base_pre_euler3d_097K.csv'
				]
	baseNPB = [ 'base_runs/base_bt_A.csv', 'base_runs/base_ft_B.csv', 'base_runs/base_sp_A.csv', 
				'base_runs/base_cg_B.csv', 'base_runs/base_lu_A.csv', 'base_runs/base_mg_C.csv'
				]


	# inRodinia = [("rodinia/kmeans/kmeans.csv",63492.0), ("rodinia/cfd/cfd_097K.csv", 157347.4),
	# 			("rodinia/sradv1/srad_v1.csv", 103462.0), ("rodinia/lud/lud_8000.csv", 350950.0),
	# 			("rodinia/lavaMD/lavaMD_20.csv", 14720.0), ("rodinia/hotspot/hotspot_1024.csv", 3144.5),
	# 			("rodinia/sradv2/srad_v2_6000.csv",151200.0), ("rodinia/pre_euler/pre_euler3d_097K.csv",168371.0)]
	# labelsRodinia = [f.split('/')[1] for (f,_) in inRodinia]

	# inNPB = [	("NPB/bt_A.csv",168300.0), ("NPB/ft_B.csv",92050.0), ("NPB/sp_A.csv",85000.0), ("NPB/cg_B.csv",54700.0), 
	# 			("NPB/lu_A.csv",119280.0), ("NPB/mg_C.csv",155700.0)]
	# labelsNPB = [f.split('/')[1].split('.')[0] for (f,_) in inNPB]


	infiles = inRodinia+inNPB
	labels = labelsRodinia+labelsNPB
	base_files = baseRod+baseNPB

	data_set = []
	for infile in zip(infiles,base_files):
		data_set.append(get_rating(infile[0][0],infile[0][1]))#/normalizer(infile[1],infile[0][1]))


	u,s,v = np.linalg.svd(data_set,full_matrices=False)
	print s

	N=len(data_set)
	M=len(data_set[0])
	K=len(s)

	u = u[:,:K]
	s = s[:K]
	v = v[:K]

	# E = sum(i**2 for i in s)
	# for i in xrange(K):
	# 	print "Energy percentage %.2f \n" % (sum(s[j]**2 for j in xrange(i+1))/E * 100.0)

	U = np.dot(u,np.diag(s**0.5))
	V = np.dot(np.diag(s**0.5),v)


	# for i in xrange(N):
	# 	print "Coords "+str(labels[i])+": (x,y,z) = (%.3f,%.3f,%.3f)" % (U[i][0],U[i][1],U[i][2])

	sim1 = np.empty((N,N))
	for i in xrange(N):
		for j in xrange(N):
			sim1[i,j] = np.dot(U[i],U[j])/(np.linalg.norm(U[i]) * np.linalg.norm(U[j]))
	
	# print "Cosine similarity"
	# for i in xrange(N):
	# 	for j in xrange(N):
	# 		print " %.2f" % sim1[i,j],
	# 	print

	# for i in xrange(N):
	# 	print labels[i]
	# 	for j in xrange(N):
	# 		if sim1[i][j] >= 0.79:
	# 			print "%s, " % labels[j],
	# 	print "\n-----------------"

	sim2 = np.empty((N,N))
	avg = np.empty(N)
	for i in xrange(N):
		avg[i] = sum(U[i])/len(U[i])
	for i in xrange(N):
		a = U[i] - avg[i]
		for j in xrange(N):
			b = U[j] - avg[j]
			sim2[i,j] = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

	# print "Pearson similarity"
	# for i in xrange(N):
	# 	for j in xrange(N):
	# 		print " %.2f" % sim2[i,j],
	# 	print

	# for i in xrange(N):
	# 	print labels[i]
	# 	for j in xrange(N):
	# 		if sim2[i][j] >= 0.80:
	# 			print "%s, " % labels[j],
	# 	print "\n-----------------"

	''' pearson similarities between predicted configurations and actual best '''

	# nofeedback = [	[846, 1326, 1326, 1326, 2224, 1612, 1612, 1612],
	# 				[2118,968,968,968,968,2807,250,250],
	# 				[2351,1326,388,388,1612,388,388,388],
	# 				[222,1326,1321,960,960,240,240,240],
	# 				[282,1326,930,966,2370,966,1398,966],
	# 				[930,1321,1326,1326,966,966,1398,966]]
	# feedback = [	[846, 1326, 1326, 1326, 2296, 2224, 1612, 1612],
	# 				[2118,968,968,968,968,2807,250,250],
	# 				[2351,1326,388,2296,1612,388,1612,1612],
	# 				[222,1321,960,960,960,2382,240,240],
	# 				[282,1321,1104,1104,966,966,966,2154],
	# 				[930,930,1326,966,966,966,966,966]]

	# nofeedback = [	[2873, 2342, 938, 2414, 1694, 1694, 335, 335],
	# 				[1064, 911, 902, 1217, 344, 1424, 704, 704],
	# 				[605, 911, 911, 947, 1271, 983, 1343, 1343],
	# 				[2117, 2387, 1271, 1235, 1235, 1199, 1199, 1190],
	# 				[424, 938, 938, 938, 200, 1010, 488, 1028],
	# 				[434, 1625, 938, 938, 218, 218, 218, 218]]
	# feedback = [	[2873, 2342, 938, 938, 938, 1046, 335, 335],
	# 				[1064, 902, 911, 1064, 1208, 704, 704, 344],
	# 				[605, 911, 902, 1271, 947, 1271, 983, 1343],
	# 				[2117, 2387, 2387, 1271, 1235, 1199, 1190, 1190],
	# 				[424, 902, 2414, 1154, 1154, 1064, 1064, 2495],
	# 				[434, 938, 2342, 938, 938, 218, 191, 218]]


	nofeedback = [	[2369, 1343, 1235, 1415, 1235, 1235, 1235, 1235],
					[1649, 983, 1343, 1226, 1226, 614, 326, 326],
					[938, 1343, 1235, 1226, 1226, 1226, 560, 1226],
					[2414, 1343, 1343, 1343, 2873, 1424, 1226, 1298],
					[1082, 974, 938, 983, 911, 911, 911, 911],
					[424, 1019, 983, 1235, 1226, 1226, 326, 1226]
				]
				
	feedback = [	[2369, 1019, 1343, 1343, 1235, 1235, 1235, 1235],
					[1649, 974, 1343, 1046, 326, 326, 614, 614],
					[938, 1019, 1235, 1226, 560, 560, 1226, 1226],
					[2414, 1235, 1235, 1415, 2846, 1433, 1370, 1424],
					[1082, 938, 1019, 911, 911, 911, 947, 911],
					[424, 1046, 1055, 1010, 1226, 1226, 326, 326]
				]


	V = V.T
	
	print 'Correlation of predicted configurations with actual - No feedback'
	sim_app = []
	for app in nofeedback:
		avg_best = sum(V[app[0]+1])/2880.0
		a = np.array(V[app[0]+1]) - avg_best
		for i in xrange(len(app)):
			avg = sum(V[app[i]+1])/2880.0
			b = np.array(V[app[i]+1]) - avg 
			sim = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
			sim_app.append(sim)
			print "%f " % sim,
		print
	tmp = np.zeros(8)
	for i in xrange(len(sim_app)):
		tmp[i%8] += sim_app[i]
	print tmp / ((i+1)/8)

	sim_app = []
	print 'Correlation of predicted configurations with actual - feedback'
	for app in feedback:
		avg_best = sum(V[app[0]+1])/2880.0
		a = np.array(V[app[0]+1]) - avg_best
		for i in xrange(len(app)):
			avg = sum(V[app[i]+1])/2880.0
			b = np.array(V[app[i]+1]) - avg 
			sim = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
			sim_app.append(sim)
			print "%f " % sim,
		print
	tmp = np.zeros(8)
	for i in xrange(len(sim_app)):
		tmp[i%8] += sim_app[i]
	print tmp / ((i+1)/8)


	# label = [(i,labels[i]) for i in range(N)]
	# fig = plt.figure()
	# ax = plt.subplot(111)
	# for p in xrange(N):
	# 	ax.scatter(U[p][0],U[p][1],label=label[p])

	# labels = [i for i in range(N)]

	# for label,x,y in zip(labels,U[:,0],U[:,1]):
	# 	plt.annotate(label,xy=(x,y), xytext=(-10,5),textcoords = 'offset points')

	# box = ax.get_position()
	# ax.set_position([box.x0,box.y0+box.height*0.1,box.width,box.height*0.9])
	# ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.05),ncol=5,scatterpoints=1)
	# ax.set_title("2D - Application Projection\nRating: MFlops/sec (Normalized to the base run)")
	# ax.grid(True)

	# sns.set_style("whitegrid")

	# fig2 = plt.figure()
	# ax2 = fig2.add_subplot(111, projection='3d')
	# for p in xrange(N):
	# 	ax2.scatter(U[p][0],U[p][1],U[p][2])
	# 	ax2.text(U[p][0],U[p][1],U[p][2],labels[p])

	# ax2.set_xlabel("Feature 1")
	# ax2.set_ylabel("Feature 2")
	# ax2.set_zlabel("Feature 3")

	# ax2.set_title("3D - Application Projection")
	# ax2.grid(True)

	# plt.show()
	# outfile = 'images/2D_app_Mflops_sec_norm.png'
	# fig.savefig(outfile,bbox_inches="tight",format='png')


	
