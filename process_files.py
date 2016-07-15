import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def get_avg_time(infile):
	tmp = 0.0
	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		for row in myreader:
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
			tmp += rtime
	return (tmp/2880)



inputfiles = [	"rodinia/hotspot3D/hotspot3D_512_4.csv", "rodinia/lavaMD/lavaMD_20.csv",
				"rodinia/myocyte/myocyte_30_228.csv", "NPB/ft_B.csv",
				"NPB/sp_A.csv", "NPB/lu_A.csv"]

labels=['HS3D', 'lvMD', 'MC', 'ft', 'sp', 'lu']

times = np.empty(len(inputfiles))

for (i,file) in enumerate(inputfiles):
	times[i] = get_avg_time(file)


pred1 = [[0.8206, 0.9689, 0.9643, 0.9631, 0.9706, 0.9857],[0.8206, 0.9689, 0.9643, 0.9911, 0.9416, 0.9857]]
pred2 = [[0.9072, 0.9689, 0.9709, 0.9911, 0.8904, 0.9817],[0.9204, 0.9689, 0.9709, 0.9911, 0.9578, 0.9817]]
pred3 = [1.0 for i in xrange(len(pred1[0]))]

times1 = times*4*5.76/60
times2 = times*4*28.8/60
times3 = times*4*2880/60

print times1
print times2
print times3
raw_input()

''' Plotting '''
ax = plt.subplot(111)
ax.set_title('Performance vs. Time')
ax.set_ylabel('Performance (normalized)')
ax.set_xlabel('Time (minutes)')
ax.set_xscale('log',base=10)
ax.grid(True,which='minor',axis='both')
# ax.scatter(times1,pred1[0],label='0.2% no feedback',marker='D',color='b')
ax.scatter(times1,pred1[1],label='0.2% feedback',marker='o',color='g')
# ax.scatter(times2,pred2[0],label='1% no feedback',marker='s',color='g')
ax.scatter(times2,pred2[1],label='1% feedback',marker='^',color='purple')
ax.scatter(times3,pred3,label='brute force search',color='r',marker='x')


for label,x,y in zip(range(1,len(times1)+1),times1,pred1[1]):
	ax.annotate(label,xy=(x,y), xytext=(-5,-15),textcoords = 'offset points')

for label,x,y in zip(range(1,len(times1)+1),times2,pred2[1]):
	ax.annotate(label,xy=(x,y), xytext=(-5,5),textcoords = 'offset points')

for label,x,y in zip(range(1,len(times1)+1),times3,pred3):
	ax.annotate(label,xy=(x,y), xytext=(0,-15),textcoords = 'offset points')


ax.legend(loc='lower right',frameon=True)

# plt.show()
outfile = 'images/perf_vs_time_2.png'
plt.savefig(outfile,bbox_inches="tight",format='png')
