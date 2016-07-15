#!/usr/bin/python

import csv
import sys
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.ioff()

x_Axis = []
cpi_Axis = []
Power_Axis = []
Temp_Axis = []
BW_Axis = []

if len(sys.argv) < 2:
	print "Give the csv input file."
	sys.exit(1)

with open(sys.argv[1],"r") as csvfile:
	myreader = csv.reader(csvfile,delimiter=',')
	myreader.next()
	for row in myreader:
		cpi_Axis.append(row[2])
		Power_Axis.append(row[6])
		Temp_Axis.append(row[7])
		BW_Axis.append(row[3])

num = len(cpi_Axis)
x_Axis = range(1,num+1)


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
ax1.plot(x_Axis, cpi_Axis,'b',marker='x')
ax1.set_title('CPI')
ax2.plot(x_Axis, BW_Axis,'b',marker='x')
ax2.set_title('BW')
ax3.plot(x_Axis,Power_Axis,'b',marker='x')
ax3.set_title('Power')
ax4.plot(x_Axis, Temp_Axis,'b',marker='x')
ax4.set_title('Temp')
plt.savefig("cfd1.png")

f2,ax11 = plt.subplots()
ax11.plot(x_Axis, cpi_Axis,'-',marker='x')
ax11.set_title('CPI')
plt.show()

#raw_input()


# fig,ax = plt.subplots()
# ax.grid(True)
# ax.set_xlabel("$Configurations$")

# x = np.arange(len(x_Axis))
# ax.set_ylabel("$BW$")
# line = ax.plot(x,BW_Axis, label="cpi", color="blue",marker='x')


# plt.title("Bandwidth")
# lgd = plt.legend(line, "upper right")
# lgd.draw_frame(False)
# plt.savefig("bw.png",bbox_inches="tight")



