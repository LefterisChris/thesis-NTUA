#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as tck

def getvalues(inputfile,period):
	l = np.zeros(period)
	for i,line in enumerate(inputfile):
		l[i%period] += float(line.split(' ')[2])
	l = l / ((i+1)/period)
	# return (sum(l)/len(l))
	return l


N = 14
# testfiles = ['backprop','lavaMD','heartwall','myocyte','sp','lu']
testfiles = ['hotspot3D', 'lavaMD', 'myocyte', 'ft', 'sp', 'lu']

period = len(testfiles)

nofeedback = []
for r in xrange(1,N+1):
	with open("evaluation/UVdec_"+str(r)+"_0.out") as infile:
		nofeedback.append(getvalues(infile,period))
nofeedback = np.array(nofeedback).T


feedback = []
for r in xrange(1,N+1):
	with open("evaluation/UVdec_"+str(r)+"_1.out") as infile:
		feedback.append(getvalues(infile,period))
feedback = np.array(feedback).T

# ts_set = [0.01,0.02,0.05,0.1,0.2,0.5]

# vals = []
# for ts in ts_set:
# 	tmp = []
# 	for r in xrange(4,8+1):
# 		with open("evaluation/UVdec_"+str(r)+"_"+str(ts)+"_0.out",'rb') as infile:
# 			tmp.append(getvalues(infile,period))
# 	vals.append(tmp)

# vals2 = []

# for ts in ts_set:
# 	tmp = []
# 	for r in xrange(4,8+1):
# 		with open("evaluation/UVdec_"+str(r)+"_"+str(ts)+"_1.out",'rb') as infile:
# 			tmp.append(getvalues(infile,period))
# 	vals2.append(tmp)

# print vals[3]
# print vals2[3]

# raw_input()

''' ploting '''
fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.grid(True)
ax1.set_title("Training size 10% without feedback")
ax1.set_ylabel("RMSE")
ax1.set_xlim((1,N))
# ax1.set_ylim((0.0,0.8))

for i in xrange(len(testfiles)):
	ax1.plot(range(1,N+1),nofeedback[i])

# ax2 = plt.subplot(111)
ax2.grid(True)
ax2.set_title("Training size 10% with feedback")
ax2.set_xlabel("Number of features")
ax2.set_ylabel("RMSE")
# ax2.set_ylim((0.0,0.8))
for i in xrange(len(testfiles)):
	ax2.plot(range(1,N+1),feedback[i],label=testfiles[i])

box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
box = ax2.get_position()
ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

ax2.legend(loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=3)

# fig.legend(tuple(lfb),tuple(testfiles),'upper right',ncol=1)

# fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
# ax1.get_yaxis().set_minor_locator(tck.AutoMinorLocator())
# ax1.grid(b=True,which='major',axis='both')
# ax1.grid(b=True,which='both',axis='y')
# ax1.set_title('RMSE exploration without feedback')
# ax1.set_ylabel('RMSE')
# # ax1.set_xlabel('Number of features')
# for i in xrange(len(vals)):
# 	ax1.plot(range(4,8+1),vals[i],label="Training size "+str(ts_set[i]*100)+"%")

# ax2.get_yaxis().set_minor_locator(tck.AutoMinorLocator())
# ax2.grid(b=True,which='major',axis='both')
# ax2.grid(b=True,which='both',axis='y')
# ax2.set_title('RMSE exploration with feedback')
# ax2.set_ylabel('RMSE')
# ax2.set_xlabel('Number of features')
# for i in xrange(len(vals)):
# 	ax2.plot(range(4,8+1),vals2[i],label="Training size "+str(ts_set[i]*100)+"0%")

# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

# ax2.legend(loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=3)

# plt.show()
outfile = 'images/rmse_feat_mflops_sec.png'
plt.savefig(outfile,bbox_inches="tight",format='png')




