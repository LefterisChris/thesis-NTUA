import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import seaborn as sns

def get_coords(infile,Mflops):
	with open(infile,"rb") as csvfile:
		myreader = csv.reader(csvfile,delimiter=',')
		myreader.next()
		row = myreader.next()
		""" 
		Configuration,grade1,CPI,BW/s((GB/sec),GFLOPS/s,Vectorization,Power(W),Temperature(C),Time(sec)
		initial test: take CPI 
		"""
		cpi = float(row[2])
		power = float(row[6])
		bw = float(row[3])
		flops = float(row[4])
		rtime = sum(map(float,row[8].strip('[]').split(','))) / 4.0
		perf = (Mflops/1000.0)/rtime
		op_int =  perf / bw
	return (op_int,perf)

def myLogFormat(y,pos):
	# Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log2(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)

x = np.arange(0.0625,32,0.1)
ridge_point_th = 4.18
ridge_point_ac = 5.67

BW_th = np.poly1d([240,0])
P_th = np.poly1d(1003.2)
BW_ac = np.poly1d([128.31,0])
P_ac = np.poly1d(727.9911)

part11 = filter(lambda y: y <= ridge_point_th,x)
part12 = list(set(x) - set(part11))
part21 = filter(lambda y: y <= ridge_point_ac,x)
part22 = list(set(x) - set(part21))
y1 = np.concatenate((BW_th(part11),P_th(part12)))
y2 = np.concatenate((BW_ac(part21),P_ac(part22)))

ax1 = plt.subplot(111)
ax1.set_xscale('log',basex=2)
ax1.set_yscale('log',basey=2)
ax1.xaxis.set_major_formatter(tck.FuncFormatter(myLogFormat))
ax1.yaxis.set_major_formatter(tck.FuncFormatter(myLogFormat))
ax1.grid(True,which='major',axis='both')
ax1.grid(True,which='minor',axis='both')
mini = 1/128.0#min(y2) - 2
maxi = 2*max(y1) 
ax1.set_ylim([mini,maxi])
ax1.set_title("Roofline model (DP)")
ax1.set_xlabel("Operational Intensity (Flops/Byte)")
ax1.set_ylabel("GFlops/sec")

ax1.annotate('1003.2 GFlops/s',xy=(ridge_point_th,max(y1)),xytext=(ridge_point_th,max(y1)+32))
ax1.annotate('727.9911 GFlops/s',xy=(ridge_point_ac,max(y2)),xytext=(ridge_point_ac,max(y2)+32))
ax1.annotate('240 GB/s',xy=(0.5,128),xytext=(0.5,240),rotation=40)
ax1.annotate('128.31 GB/s',xy=(0.5,64),xytext=(0.5,160),rotation=40)

ax1.vlines(x=ridge_point_th,ymin=mini,ymax=max(y1),color='red',linestyle='dashed')
ax1.vlines(x=ridge_point_ac,ymin=mini,ymax=max(y2),color='red',linestyle='dashed')

ax1.plot(x,y1,label="Theoretical")
ax1.plot(x,y2,label="Achievable")
ax1.legend(loc='upper left')

################################################
# 				Applications				   #
################################################

inNPB = [	("NPB/base_bt_A.csv",168300.0), ("NPB/base_ft_B.csv",92050.0), ("NPB/base_sp_A.csv",85000.0), ("NPB/base_cg_B.csv",54700.0), 
			("NPB/base_lu_A.csv",119280.0), ("NPB/base_mg_C.csv",155700.0)]

labelsNPB = [f.split('/')[1].split('.')[0] for (f,_) in inNPB]

for infile in inNPB:
	x,y = get_coords(infile[0],infile[1])
	print x,y 
	raw_input()
	ax1.scatter([x],[y],color='red')

plt.show()

#plt.savefig("roofline/my_roofline_dp.png",bbox_inches="tight")