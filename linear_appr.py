import numpy as np
import math

def calc_error(xlist,ylist,pol1,pol2,limit):
	err1 = 0.0
	err2 = 0.0
	end = len(xlist)-1
	for i in xrange(limit+1):
		err1 += abs(ylist[i]-pol1(xlist[i]))
	err1 = (err1/limit)
	for i in xrange(limit,end):
		err2 += abs(ylist[i]-pol2(xlist[i]))
	err2 = (err2/(end-limit+1))
	return err1+err2


density = []
flops = []
bw = []
with open("roofline/results.out") as rl_file:
	rl_file.readline()
	for line in rl_file:
		line = line.split()
		density.append(float(line[1]))
		flops.append(float(line[2]))
		bw.append(float(line[3]))

x = np.array(density[:50])
y = np.array(flops[:50])

N = len(x)

min_c = x[0]
min_error = 10000000.0

for i in xrange(1,N-1):
	a1 = (y[i] - y[0])/(x[i]-x[0])
	a2 = (x[i]*y[0] - x[0]*y[i]) / (x[i]-x[0])
	p1 = np.poly1d([a1,a2])

	b1 = (y[N-1] - y[i]) / (x[N-1] - x[i])
	b2 = (x[N-1]*y[i] - x[i]*y[N-1]) / (x[N-1] - x[i])
	p2 = np.poly1d([b1,b2])

	error = calc_error(x,y,p1,p2,i)
	print i,error
	if error < min_error:
		min_c = i
		min_error = error

print min_c

print "(%.2f,%.3f)" % (x[min_c],y[min_c])