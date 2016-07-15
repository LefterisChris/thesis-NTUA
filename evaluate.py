#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import csv
import seaborn as sns

def get_base_rating(infile):
	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		row = myreader.next()
		""" 
		Configuration,grade1,CPI,BW/s((GB/sec),GFLOPS/s,Vectorization,Power(W),Temperature(C),Time(sec)
		"""
		threads = 4
		cores = 57
		cpi = float(row[2])
		power = float(row[6])
		bw = float(row[3])
		flops = float(row[4])
		rtime = sum(map(float,row[8].strip('[]').split(','))) / 4.0
		# rating = 1000.0* threads / (cpi * power)
		rating = power*rtime
	return rating

def get_permutation():
	permutation = []

	libhugetlbfs = ["","LD_PRELOAD=/home/echristof/hugetlbfs/obj64/libhugetlbfs.so HUGETLB_MORECORE=yes "]
	opts = ["-O2 ", "-O3 "]
	prefetch = ["-opt-prefetch=0 ","-opt-prefetch=2 ","-opt-prefetch=3 ","-opt-prefetch=4 "]
	streaming_stores1 = ["-opt-streaming-stores never ","-opt-streaming-stores always "]
	streaming_stores2 = ["-opt-streaming-cache-evict=0 ","-opt-streaming-cache-evict=1 ",
					"-opt-streaming-cache-evict=2 ","-opt-streaming-cache-evict=3 "]
	unroll = ["-unroll=0 ","-unroll "]

	for f1 in opts:
		for env1 in libhugetlbfs:
			for f2 in prefetch:
				for f3 in streaming_stores1:
					if f3 == "-opt-streaming-stores never ":
						f4 = ""
						for f5 in unroll:
							for affinity in ['scatter','balanced']:
								for cores in [19,38,57]:
									for threads in [2,3,4]:
										permutation.append([f1,f2,f3,f4,f5,env1,str(cores),str(threads),affinity])
					else:
						for f4 in streaming_stores2:
							for f5 in unroll:
								for affinity in ['scatter','balanced']:
									for cores in [19,38,57]:
										for threads in [2,3,4]:
											permutation.append([f1,f2,f3,f4,f5,env1,str(cores),str(threads),affinity])
	return permutation

def normalizer(infile, Mflops):
	with open(infile,'rb') as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		line = myreader.next()
		power = float(line[6])
		rtime = sum(map(float,line[8].strip('[]').split(','))) / 4.0
	return (10.0*(Mflops/rtime)/power)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height,
                '%.4f' % height, ha='center', va='bottom')

# pre = 'UVdec_'
# inputfiles2 = ['sradv2', 'cfd', 'streamcluster', 'nn', 'lu', 'mg']
# inputfiles = [pre+f for f in inputfiles2]

# pre='base_runs/'
# base_files = ['base_srad_v2_6000.csv', 'base_euler3d_097K.csv','base_streamcluster.csv', 
# 				'base_nn_filelist_1G.csv', 'base_lu_A.csv', 'base_mg_C.csv']
# base_files = [pre+f for f in base_files]


# pre = 'UVdec_'
# inputfiles2 = ['hotspot3D', 'lavaMD', 'myocyte', 'ft', 'sp', 'lu']
# inputfiles = [pre+f for f in inputfiles2]

# pre='base_runs/'
# base_files = ['base_hotspot3D_512_4.csv', 'base_lavaMD.csv','base_myocyte_30_228.csv', 
# 				'base_ft_B.csv','base_sp_A.csv', 'base_lu_A.csv']
# base_files = [pre+f for f in base_files]

pre = 'UVdec_'
inputfiles2 = ['backprop', 'lavaMD', 'heartwall', 'myocyte', 'sp', 'lu']
inputfiles = [pre+f for f in inputfiles2]

pre='base_runs/'
base_files = ['base_backprop_4194304.csv', 'base_lavaMD.csv','base_heartwall_30.csv' ,'base_myocyte_30_228.csv', 
				'base_sp_A.csv', 'base_lu_A.csv']
base_files = [pre+f for f in base_files]

files = zip(inputfiles,base_files)

configuration = get_permutation() 

TRAIN_SET = ['0.002']
values=[]
for (f1,f2) in files:
	max_val = np.zeros(len(TRAIN_SET)+1)
	max_idx = np.zeros(len(TRAIN_SET)+1,dtype=int)
	predicted = np.zeros((len(TRAIN_SET),2880))
	tmp = []
	for (j,ts) in enumerate(TRAIN_SET):
		actual = np.zeros(2880)
		with open("evaluation/lb2/"+f1+"_"+ts+"_0.csv") as fp:
			for i,line in enumerate(fp):
				tokens = line.split(',')
				actual[i%2880] += float(tokens[1])
				predicted[j][i%2880] += float(tokens[2])
			actual = actual / ((i+1)/2880)
			predicted[j] = predicted[j] / ((i+1)/2880)

		for i in xrange(2880):
			if max_val[0] < actual[i]:
				max_val[0] = actual[i]
				max_idx[0] = i
			
			if max_val[j+1] < predicted[j][i]:
				max_val[j+1] = predicted[j][i]
				max_idx[j+1] = i

	base_rating = 10#get_base_rating(f2)

	print "TEST: %s" % f1
	# print "Count of maxes: actuals=%d predicted=%d" %(cnt_max1,cnt_max2)
	# print "Actual max: index=%d value=%f" % (max_idx1,max_val1)
	# print "Predicted max: index=%d value=%f" %(max_idx2,max_val2)
	# print "Actual value from prediction: %f" % actual[max_idx2]	
	# print "Degradation in performance: %f%%" % ((actual[max_idx2]/max_val1 - 1.0)*100.0)
	# print "Change from base rating: %.2f-fold improvement." % (actual[max_idx2]/base_rating)
	# print "--------------------------------\n"
	# print "%f,%f,%f,%f,%f" % (max_val[0],actual[max_idx[1]],actual[max_idx[2]],actual[max_idx[3]],base_rating)
	print 'Actual conf: %s ' % configuration[max_idx[0]]
	print 'Predicted 0.2%% conf: %s ' % configuration[max_idx[1]]
	# print 'Predicted 0.2%% conf: %s ' % configuration[max_idx[2]]
	# print 'Predicted 1%% conf: %s ' % configuration[max_idx[3]]
	# print 'Predicted 5%% conf: %s ' % configuration[max_idx[4]]
	val_vector = np.array([max_val[0],actual[max_idx[1]],base_rating])/max_val[0]
	values.append(val_vector)
	print "--------------------------------\n"

values = np.array(values).T

''' plotting '''
ax = plt.subplot(111)

width = 0.50
opacity=0.5

x = np.arange(len(inputfiles2))
# rects1 = ax.bar(x-width,values[0],width,color='b',alpha=opacity,label='actual')
rects2 = ax.bar(x,values[1],width,color='g',alpha=opacity,label='predicted')
# rects3 = ax.bar(x+width,values[2],width,color='r',alpha=opacity,label='base')
# rects4 = ax.bar(x+width,values[3],width,color='r',alpha=opacity, label='predicted 1%') 
# rects5 = ax.bar(x+width,values[4],width,color='purple',alpha=opacity, label='predicted 5%')
# rects6 = ax.bar(x+2*width,values[4],width,color='purple',alpha=opacity,label='base')	# base

autolabel(rects2)
ax.set_ylim([0.0,1.05])

ax.set_ylabel('Rating')
ax.set_title('Performance of predicted tuning configurations')
ax.set_xticks(x+width/2)
ax.set_xticklabels(tuple(inputfiles2))

# ax.legend(loc='upper right',frameon=True)

# plt.show()
outfile = 'images/perf_comp_intro.png'
plt.savefig(outfile,bbox_inches="tight",format='png')

# TRAIN_SET = ['0.001','0.002','0.005','0.01','0.02','0.05','0.10']
# nofeedback = []
# feedback = []

# for (f1,f2) in files:
# 	predicted = np.zeros((len(TRAIN_SET),2880))
# 	max_val = np.zeros(len(TRAIN_SET)+1)
# 	max_idx = np.zeros(len(TRAIN_SET)+1,dtype=int)
# 	tmp = []
# 	for (j,ts) in enumerate(TRAIN_SET):
# 		with open("evaluation/flops_power/"+f1+"_"+ts+"_0.csv") as fp:
# 			actual = np.zeros(2880)
# 			for i,line in enumerate(fp):
# 				tokens = line.split(',')
# 				actual[i%2880] += float(tokens[1])
# 				predicted[j][i%2880] += float(tokens[2])
# 			actual = actual / ((i+1)/2880)
# 			predicted[j] = predicted[j] / ((i+1)/2880)

# 		for i in xrange(2880):
# 			if max_val[0] < actual[i]:
# 				max_val[0] = actual[i]
# 				max_idx[0] = i
			
# 			if max_val[j+1] < predicted[j][i]:
# 				max_val[j+1] = predicted[j][i]
# 				max_idx[j+1] = i

# 		base_rating = get_base_rating(f2)

# 	tmp.append(max_val[0])
# 	for i in max_idx[1:]:
# 		tmp.append(actual[i]) 
# 	# tmp.append(base_rating)
	
# 	nofeedback.append(np.array(tmp))

# 	print "No feedback"
# 	print max_idx

# 	tmp = []
# 	predicted = np.zeros((len(TRAIN_SET),2880))
# 	max_val = np.zeros(len(TRAIN_SET)+1)
# 	max_idx = np.zeros(len(TRAIN_SET)+1,dtype=int)
# 	for (j,ts) in enumerate(TRAIN_SET):
# 		with open("evaluation/flops_power/"+f1+"_"+ts+"_1.csv") as fp:
# 			actual = np.zeros(2880)
# 			for i,line in enumerate(fp):
# 				tokens = line.split(',')
# 				actual[i%2880] += float(tokens[1])
# 				predicted[j][i%2880] += float(tokens[2])
# 			actual = actual / ((i+1)/2880)
# 			predicted[j] = predicted[j] / ((i+1)/2880)

# 		for i in xrange(2880):
# 			if max_val[0] < actual[i]:
# 				max_val[0] = actual[i]
# 				max_idx[0] = i
			
# 			if max_val[j+1] < predicted[j][i]:
# 				max_val[j+1] = predicted[j][i]
# 				max_idx[j+1] = i

# 		base_rating = get_base_rating(f2)


# 	tmp.append(max_val[0])
# 	for i in max_idx[1:]:
# 		tmp.append(actual[i]) 
# 	# tmp.append(base_rating)
	
# 	feedback.append(np.array(tmp))

# 	print 'Feedback'
# 	print max_idx


# norm_nofeedback = []

# for vec in nofeedback:
# 	tmp = vec[1:]/vec[0]
# 	norm_nofeedback.append(tmp)

# norm_nofeedback = np.array(norm_nofeedback).T
# print "Normalized no feedback"
# print norm_nofeedback

# avg_sums = [sum(vec)/len(vec) for vec in norm_nofeedback]

# norm_feedback = []
# for vec in feedback:
# 	tmp = vec[1:]/vec[0]
# 	norm_feedback.append(tmp)

# norm_feedback = np.array(norm_feedback).T
# print "Normalized feedback"
# print norm_feedback

# avg_sums_fb = [sum(vec)/len(vec) for vec in norm_feedback]


# ''' Plotting '''
# ax = plt.subplot(111)

# width = 0.6
# opacity=0.5

# x = np.arange(len(avg_sums))*2
# # x = range(len(avg_sums))
# rect1 = ax.bar(x,avg_sums,width=width,alpha=opacity,color='g',label='no feedback')
# rect2 = ax.bar(x+width,avg_sums_fb,width=width,alpha=opacity,color='y',label='feedback')

# ax.set_ylabel('Rating Normalized')
# ax.set_xlabel('Training Sizes for the Incoming Apps')
# ax.set_title('Comparison of Predicted Ratings')

# ax.set_xticks(x+width)
# # TRAIN_SET.append('base')
# xlabels = [str(float(i)*100.0)+"%" for i in TRAIN_SET]
# ax.set_xticklabels(tuple(xlabels))
# ax.legend(loc='lower right',frameon=True)

# autolabel(rect1)
# autolabel(rect2)

# plt.tight_layout()
# plt.show()
# outfile = 'images/perf_comp_size_ipc_pw.png'
# plt.savefig(outfile,bbox_inches="tight",format='png')



