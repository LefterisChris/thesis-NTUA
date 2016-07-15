import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(infile,Mflops,parameters):
	out_array = [[] for i in xrange(len(parameters))]
	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		for row in myreader:
			""" 
			Configuration,grade1,CPI,BW/s((GB/sec),GFLOPS/s,Vectorization,Power(W),Temperature(C),Time(sec)
			initial test: take CPI 
			"""
			conf = row[0][1:-1].split(",")
			opt = conf[0][1:-2]
			prefetch = conf[1][2:-2]
			sstores = conf[2][2:-2]
			cache_evict = conf[3][2:-2]
			unroll = conf[4][2:-2]
			huge_pages = conf[5]
			threads = int(conf[7][2:-1])
			cores = int(conf[6][2:-1])
			affinity = conf[8][2:-1]
			
			cpi = float(row[2])
			power = float(row[6])
			bw = float(row[3])
			flops = float(row[4])
			rtime = sum(map(float,row[8].strip('[]').split(','))) / 4.0
			# ratings = 10.0* cores * threads / (cpi * power)
			ratings = ((Mflops/rtime)/power)
			#ratings[i] = 10000.0*(flops[i]/bw[i])/power[i]
			#ratings[i] = flops[i]
			#ratings[i] = flops[i]*rtime[i] / power[i]
			for i,p in enumerate(parameters):
					# if p in affinity:
					# if cores == p[0] and threads == p[1]:
					# if p in huge_pages:
					# if p in unroll:
					# if p in sstores: 
					# 	out_array[i].append(ratings)
					# 	break
					if p in opt:
						out_array[i].append(ratings)
						break

	return np.array(out_array)

def normalizer(infile, Mflops):
	with open(infile,'rb') as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		line = myreader.next()
		power = float(line[6])
		rtime = sum(map(float,line[8].strip('[]').split(','))) / 4.0
		# print power, rtime, Mflops
	return ((Mflops/rtime)/power)
	

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
			'base_runs/base_srad_v2_6000.csv', 'base_runs/base_pre_euler3d_097K.csv',
			]
baseNPB = [ 'base_runs/base_bt_A.csv', 'base_runs/base_ft_B.csv', 'base_runs/base_sp_A.csv', 
			'base_runs/base_cg_B.csv', 'base_runs/base_lu_A.csv', 'base_runs/base_mg_C.csv'
			]


#	Affinity type
# parameters = ['scatter','balanced']

infiles = inRodinia+inNPB
base_files = baseRod+baseNPB

#	Cores and Threads
# p1 = [19,38,57]
# p2 = [2,3,4]
# goo = lambda x,y: [] if y == [] else goo(x,y[1:])+[(x,y[0])]
# foo = lambda x,y: [] if x == [] else foo(x[1:],y)+goo(x[0],y)
# parameters = foo(p1,p2)[::-1]
# parameters=[]
# for i in parameters1:
# 	for j in parameters2:
# 		parameters.append((i,j))

#	Huge Pages
# parameters = ["libhugetlbfs.so", ""]

#	unroll
# parameters = ["-unroll=0","-unroll"]

#	cache evict
# parameters = ['never','0','1','2','3']

#	streaming stores
# parameters = ["never", "always"]

#	prefetch
# parameters = ['0','2','3','4']

#	optimization
parameters = ['2','3']

data = []
for infile in zip(inRodinia,baseRod):
	# tmp = get_data(infile[0][0],infile[0][1],parameters)
	# tmp = np.array(tmp)
	# tmp = tmp / normalizer(infile[1], infile[0][1])# np.max(tmp)
	# tmp = tmp.tolist()
	# data = map(lambda x,y: x if y == None else y if x == None else x+y, data, tmp)

	data = get_data(infile[0][0],infile[0][1],parameters)/normalizer(infile[1], infile[0][1])
	data = data.tolist()
	for i in data:
		print len(i)


	# label = infile[0][0].split('/')[1].split('.')[0].split('_')[0]
	label = infile[0][0].split('/')[1]

	fig,ax = plt.subplots()

	sns.violinplot(data=data)
	ax.set_title("Optimization level for "+label)
	# ax.set_xlabel("")
	ax.set_ylabel("(Mflops/sec)/Watt (Normalized to base run)")


	plt.xticks(np.arange(len(data)), parameters)
	outfile = "images/violins/violin_opt_"+label+".png"
	fig.savefig(outfile,bbox_inches="tight",format='png')
	# fig.show()
	fig.clf()

data = []
for infile in zip(inNPB,baseNPB):
	# tmp = get_data(infile[0][0],infile[0][1],parameters)
	# tmp = np.array(tmp)
	# tmp = tmp / normalizer(infile[1], infile[0][1])# np.max(tmp)
	# tmp = tmp.tolist()
	# data = map(lambda x,y: x if y == None else y if x == None else x+y, data, tmp)

	data = get_data(infile[0][0],infile[0][1],parameters)/normalizer(infile[1], infile[0][1])
	data = data.tolist()
	for i in data:
		print len(i)


	label = infile[0][0].split('/')[1].split('.')[0].split('_')[0]
	# label = infile[0][0].split('/')[1]

	fig,ax = plt.subplots()

	sns.violinplot(data=data)
	ax.set_title("Optimization level for "+label)
	# ax.set_xlabel("")
	ax.set_ylabel("(Mflops/sec)/Watt (Normalized to base run)")


	plt.xticks(np.arange(len(data)), parameters)
	outfile = "images/violins/violin_opt_"+label+".png"
	fig.savefig(outfile,bbox_inches="tight",format='png')
	# fig.show()
	fig.clf()



