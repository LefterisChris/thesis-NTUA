#!/usr/bin/python


import power_logger
import subprocess
import sys
from time import sleep
import csv

import parse_likwid_base


def compiler():
	#compile program
	template = open("config/NAS.samples/make.def_intel","r")
	makefile = open("config/make.def","w")

	for i,line in enumerate(template):
		if i == 48:
			makefile.write("FFLAGS = "+compilation_flags+"\n")
		elif i == 54:
			makefile.write("FLINKFLAGS = "+compilation_flags+linking_flags+"\n")
		elif i == 84:
			makefile.write("C_LIB = -lm\n")
		elif i == 94:
			makefile.write("CFLAGS = "+compilation_flags+"\n")
		elif i == 100:
			makefile.write("CLINKFLAGS = "+compilation_flags+linking_flags+"\n")
		else:
			makefile.write(line)

	template.close()
	makefile.close()

	cmd = "make CLASS="+kernel_size+" "+kernel
	ret = subprocess.call(cmd,shell=True)
	if(ret != 0):
		print "Error compile"
		sys.exit(1)

def performer():
	# Copy the executable to mic0
	cmd = "scp bin/"+kernel_exec+" mic0:/home/echristof"
	ret = subprocess.call(cmd,shell=True)
	if(ret != 0):
		print "Error scp executable."
		sys.exit(1)

	proc_ids = "1"

	cmd = "KMP_PLACE_THREADS=01c,01t,00o"+\
		" likwid_mic/bin/likwid-perfctr -o out2.csv -c "+proc_ids+\
		" -g INSTRUCTIONS_EXECUTED:PMC0,CPU_CLK_UNHALTED:PMC1 ./"+kernel_exec

	out = "pwr.csv"
	stats = power_logger.Stats(out, 0.5)
	stats.start()

	ret = subprocess.call("ssh mic0 "+cmd,shell=True)
	if (ret != 0):
		print "Error"
		stats.shutdown()
		stats.join()
		sys.exit(1)

	sleep(3)
	stats.shutdown()
	stats.join()

	cmd = "KMP_PLACE_THREADS=01c,01t,00o"+\
		" likwid_mic/bin/likwid-perfctr -o out1.csv -c "+proc_ids+\
		" -g VPU_INSTRUCTIONS_EXECUTED:PMC0,VPU_ELEMENTS_ACTIVE:PMC1 ./"+kernel_exec

	ret = subprocess.call("ssh mic0 "+cmd,shell=True)
	if (ret != 0):
		print "Error"
		sys.exit(1)

	cmd = "KMP_PLACE_THREADS=01c,01t,00o"+\
		" likwid_mic/bin/likwid-perfctr -o out3.csv -c "+proc_ids+\
		" -g L2_DATA_READ_MISS_MEM_FILL:PMC0,DATA_CACHE_LINES_WRITTEN_BACK:PMC1 ./"+kernel_exec

	ret = subprocess.call("ssh mic0 "+cmd,shell=True)
	if (ret != 0):
		print "Error"
		sys.exit(1)

	cmd = "KMP_PLACE_THREADS=01c,01t,00o"+\
		" likwid_mic/bin/likwid-perfctr -o out4.csv -c "+proc_ids+\
		" -g L2_DATA_WRITE_MISS_MEM_FILL:PMC0 ./"+kernel_exec

	ret = subprocess.call("ssh mic0 "+cmd,shell=True)
	if (ret != 0):
		print "Error"
		sys.exit(1)


	cmd = "scp mic0:/home/echristof/*.csv ."
	if subprocess.call(cmd,shell=True) != 0:
		print "Error"
		sys.exit(1)
	
	cmd = "cat out1.csv out2.csv out3.csv out4.csv > out.csv"
	if subprocess.call(cmd,shell=True) != 0:
		print "Error"
		sys.exit(1)

	parse_likwid_base.parse("out.csv","pwr.csv","base 1 thread",out_file)

	cmd = "rm pwr.csv out*"
	if subprocess.call(cmd,shell=True) != 0:
		print "Error"
		sys.exit(1)


#main program
if len(sys.argv) < 3:
	print "Usage %s <exec> <class>\n" % sys.argv[0]
    sys.exit(1)
	
kernel = sys.argv[1]
kernel_size = sys.argv[2]
kernel_exec = kernel+"."+kernel_size+".x"
CC = "icc"

out_file = "results/base_"+kernel+"_"+kernel_size+".csv"
with open(out_file,"w+") as csvfile:
	headers = ['Configuration','grade1','CPI', 'BW/s((GB/sec)', 'GFLOPS/s', 'Vectorization', 'Power(W)', 'Temperature(C)','Time(sec)']
	writer = csv.writer(csvfile, delimiter = ',')
	writer.writerow(headers)

csvfile.close()
		
conf = [""]

compilation_flags = "-mmic -openmp -fno-alias "
linking_flags = "-Wl,-rpath=/home/echristof/libraries "

compiler()
performer()