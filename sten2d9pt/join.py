import csv
import collections
import sys

values_list = []	

headers = ['Configuration','grade1','CPI', 'BW/s((GB/sec)', 'GFLOPS/s', 'Vectorization', 'Power(W)', 'Temperature(C)','Time(sec)']

for num in range(80,int(sys.argv[1])+1):

	with open("sten2d9pt_pad_"+str(num)+".csv","rb") as csvin:
		myreader = csv.reader(csvin,delimiter=',')
		myreader.next()
		values = myreader.next()
		grade1 = float(values[3])/(float(values[0])+float(values[2])+float(values[7]))
		values = [values[6],round(grade1,5),round(float(values[0]),5),round(float(values[3]),5),round(float(values[5]),5),\
				round(float(values[4]),5), float(values[7]),float(values[2]),float(values[1])]
		values_list.append(values)


with open("sten2d9p_pad.csv","w+b") as csvout:
	writer = csv.writer(csvout, delimiter = ',')
	writer.writerow(headers)
	for l in values_list:
		writer.writerow(l)