#!/usr/bin/python


import power_logger
import subprocess
import sys
from time import sleep
import csv

import parse_likwid


def compiler():
	#compile program
	template = open("config/NAS.samples/make.def_intel","r")
	makefile = open("config/make.def","w")

	for i,line in enumerate(template):
		if i == 48:
			makefile.write("FFLAGS = "+compilation_flags+f1+f2+f3+f4+f5+"\n")
		elif i == 54:
			makefile.write("FLINKFLAGS = "+compilation_flags+linking_flags+f1+f2+f3+f4+f5+"\n")
		elif i == 84:
			makefile.write("C_LIB = -lm\n")
		elif i == 94:
			makefile.write("CFLAGS = "+compilation_flags+f1+f2+f3+f4+f5+"\n")
		elif i == 100:
			makefile.write("CLINKFLAGS = "+compilation_flags+linking_flags+f1+f2+f3+f4+f5+"\n")
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

	#Start profiling
	for affinity in ['scatter','balanced']:
		for cores in [19,38,57]:
			for threads in [2,3,4]:
				
				conf = [f1,f2,f3,f4,f5,env1,str(cores),str(threads),affinity]

				id_l = []
				for i in range(1,cores+1):
					ids = (i-1)*4 +1
					if ids+threads-1 == 228:
						id_l.append(str(ids)+"-"+str(ids+threads-2)+",0")
					else:
						id_l.append(str(ids)+"-"+str(ids+threads-1))
				proc_ids = ','.join(id_l)


				cmd = "KMP_AFFINITY="+affinity+" KMP_PLACE_THREADS="+str(cores)+"c,"+str(threads)+"t,00o"+\
					" likwid_mic/bin/likwid-perfctr -o out2.csv -c "+proc_ids+\
					" -g INSTRUCTIONS_EXECUTED:PMC0,CPU_CLK_UNHALTED:PMC1 env "+env1+"./"+kernel_exec

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

				cmd = "KMP_AFFINITY="+affinity+" KMP_PLACE_THREADS="+str(cores)+"c,"+str(threads)+"t,00o"+\
					" likwid_mic/bin/likwid-perfctr -o out1.csv -c "+proc_ids+\
					" -g VPU_INSTRUCTIONS_EXECUTED:PMC0,VPU_ELEMENTS_ACTIVE:PMC1 env "+env1+"./"+kernel_exec

				ret = subprocess.call("ssh mic0 "+cmd,shell=True)
				if (ret != 0):
					print "Error"
					sys.exit(1)

				cmd = "KMP_AFFINITY="+affinity+" KMP_PLACE_THREADS="+str(cores)+"c,"+str(threads)+"t,00o"+\
					" likwid_mic/bin/likwid-perfctr -o out3.csv -c "+proc_ids+\
					" -g L2_DATA_READ_MISS_MEM_FILL:PMC0,DATA_CACHE_LINES_WRITTEN_BACK:PMC1 env "+env1+"./"+kernel_exec

				ret = subprocess.call("ssh mic0 "+cmd,shell=True)
				if (ret != 0):
					print "Error"
					sys.exit(1)

				cmd = "KMP_AFFINITY="+affinity+" KMP_PLACE_THREADS="+str(cores)+"c,"+str(threads)+"t,00o"+\
					" likwid_mic/bin/likwid-perfctr -o out4.csv -c "+proc_ids+\
					" -g L2_DATA_WRITE_MISS_MEM_FILL:PMC0 env "+env1+"./"+kernel_exec

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


#main program

kernel = ""
kernel_size = ""
kernel_exec = kernel+"."+kernel_size+".x"
CC = "icc"

out_file = "results/"+kernel+"_"+kernel_size+".csv"
with open(out_file,"w+") as csvfile:
	headers = ['Configuration','grade1','CPI', 'BW/s((GB/sec)', 'GFLOPS/s', 'Vectorization', 'Power(W)', 'Temperature(C)','Time(sec)']
	writer = csv.writer(csvfile, delimiter = ',')
	writer.writerow(headers)

csvfile.close()
		
conf = []

compilation_flags = "-mmic -openmp -fno-alias "
linking_flags = "-Wl,-rpath=/home/echristof/libraries "
libhugetlbfs = ["","LD_PRELOAD=/home/echristof/hugetlbfs/obj64/libhugetlbfs.so HUGETLB_MORECORE=yes "]
opts = ["-O2 ", "-O3 "]
prefetch = ["-opt-prefetch=0 ","-opt-prefetch=2 ","-opt-prefetch=3 ","-opt-prefetch=4 "]
streaming_stores1 = ["-opt-streaming-stores never ","-opt-streaming-stores always "]
streaming_stores2 = ["-opt-streaming-cache-evict=0 ","-opt-streaming-cache-evict=1 ",
					"-opt-streaming-cache-evict=2 ","-opt-streaming-cache-evict=3 "]
# "-unroll-aggressive "
unroll = ["-unroll=0 ","-unroll "]


for f1 in opts:
	for env1 in libhugetlbfs:
		for f2 in prefetch:
			for f3 in streaming_stores1:
				if f3 == "-opt-streaming-stores never ":
					f4 = ""
					for f5 in unroll:
						compiler()
						performer()
				else:
					for f4 in streaming_stores2:
						for f5 in unroll:
							compiler()
							performer()
