import csv
import collections
import sys

values_list = []	

headers = ['Configuration','grade1','CPI', 'BW/s((GB/sec)', 'GFLOPS/s', 'Vectorization', 'Power(W)', 'Temperature(C)','Time(sec)']

with open("backprop_65536.csv","rb") as csvin:
		myreader = csv.reader(csvin,delimiter=',')
		myreader.next()
		for row in myreader:
			values_list.append([row[0],round(float(row[1]),4),round(float(row[2]),4),round(float(row[3]),4),round(float(row[4]),4), \
			round(float(row[5]),4), float(row[6]),float(row[7]),row[8]])


with open("backprop_65536_2.csv","w+b") as csvout:
	writer = csv.writer(csvout, delimiter = ',')
	writer.writerow(headers)
	for l in values_list:
		writer.writerow(l)
