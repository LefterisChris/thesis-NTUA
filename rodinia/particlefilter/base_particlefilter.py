#!/usr/bin/python


import power_logger
import subprocess
import sys
from time import sleep
import csv

import parse_likwid

def compiler():
	#compile program
	cmd = CC+" "+compilation_flags+"ex_particle_OPENMP_seq.c -o "+kernel_exec
	ret = subprocess.call(cmd,shell=True)
	if(ret != 0):
		print "Error compile"
		sys.exit(1)

def performer():
	# Copy the executable to mic0
	cmd = "scp "+kernel_exec+" mic0:/home/echristof"
	ret = subprocess.call(cmd,shell=True)
	if(ret != 0):
		print "Error scp executable."
		sys.exit(1)

	proc_ids = "1"

	cmd = "KMP_PLACE_THREADS=01c,01t,00o"+\
		" likwid_mic/bin/likwid-perfctr -o out2.csv -c "+proc_ids+\
		" -g INSTRUCTIONS_EXECUTED:PMC0,CPU_CLK_UNHALTED:PMC1 ./"+kernel_exec+" "+kernel_args

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
		" -g VPU_INSTRUCTIONS_EXECUTED:PMC0,VPU_ELEMENTS_ACTIVE:PMC1 ./"+kernel_exec+" "+kernel_args

	ret = subprocess.call("ssh mic0 "+cmd,shell=True)
	if (ret != 0):
		print "Error"
		sys.exit(1)

	cmd = "KMP_PLACE_THREADS=01c,01t,00o"+\
		" likwid_mic/bin/likwid-perfctr -o out3.csv -c "+proc_ids+\
		" -g L2_DATA_READ_MISS_MEM_FILL:PMC0,DATA_CACHE_LINES_WRITTEN_BACK:PMC1 ./"+kernel_exec+" "+kernel_args

	ret = subprocess.call("ssh mic0 "+cmd,shell=True)
	if (ret != 0):
		print "Error"
		sys.exit(1)

	cmd = "KMP_PLACE_THREADS=01c,01t,00o"+\
		" likwid_mic/bin/likwid-perfctr -o out4.csv -c "+proc_ids+\
		" -g L2_DATA_WRITE_MISS_MEM_FILL:PMC0 ./"+kernel_exec+" "+kernel_args

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

	parse_likwid.parse("out.csv","pwr.csv",conf,out_file)

	cmd = "rm pwr.csv out*"
	if subprocess.call(cmd,shell=True) != 0:
		print "Error"
		sys.exit(1)



kernel_exec = "particle_filter"
kernel_args = "-x 512 -y 512 -z 10 -np 80000"
CC = "icc"

out_file = "results/base_particle_filter_512_512_10_80000.csv"
with open(out_file,"w+") as csvfile:
	headers = ['Configuration','grade1','CPI', 'BW/s((GB/sec)', 'GFLOPS/s', 'Vectorization', 'Power(W)', 'Temperature(C)','Time(sec)']
	writer = csv.writer(csvfile, delimiter = ',')
	writer.writerow(headers)

csvfile.close()

compilation_flags = "-mmic -openmp -fno-alias -Wl,-rpath=/home/echristof/libraries "
conf = [" "]

compiler()
performer()