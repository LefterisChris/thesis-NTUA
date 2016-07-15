#!/usr/bin/python


import power_logger
import subprocess
import sys
from time import sleep
import csv

import parse_likwid


def compiler():
	#compile program
	cmd = CC+" "+compilation_flags+f1+f2+f3+f4+f5+" nn_openmp.c -o "+kernel_exec+" -lm"
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

	cntr2 = 1
	#Start profiling
	for affinity in ['scatter','balanced']:
		for cores in [19,38,57]:
			for threads in [2,3,4]:
				
				conf = [f1,f2,f3,f4,f5,env1,str(cores),str(threads),affinity]

				kernel_args = ' '.join(args)

				id_l = []
				for i in range(1,cores+1):
					ids = (i-1)*4 +1
					if ids+threads-1 == 228:
						id_l.append(str(ids)+"-"+str(ids+threads-2)+",0")
					else:
						id_l.append(str(ids)+"-"+str(ids+threads-1))
				proc_ids = ','.join(id_l) 
				# if cores*4 == 228:
				# 	proc_ids = "1-227,0"
				# else:
				# 	proc_ids = "1-"+str(cores*4)

				cmd = "KMP_AFFINITY="+affinity+" KMP_PLACE_THREADS="+str(cores)+"c,"+str(threads)+"t,00o"+\
					" likwid_mic/bin/likwid-perfctr -o out2.csv -c "+proc_ids+\
					" -g INSTRUCTIONS_EXECUTED:PMC0,CPU_CLK_UNHALTED:PMC1 env "+env1+"./"+kernel_exec+" "+kernel_args

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
					" -g VPU_INSTRUCTIONS_EXECUTED:PMC0,VPU_ELEMENTS_ACTIVE:PMC1 env "+env1+"./"+kernel_exec+" "+kernel_args

				ret = subprocess.call("ssh mic0 "+cmd,shell=True)
				if (ret != 0):
					print "Error"
					sys.exit(1)

				cmd = "KMP_AFFINITY="+affinity+" KMP_PLACE_THREADS="+str(cores)+"c,"+str(threads)+"t,00o"+\
					" likwid_mic/bin/likwid-perfctr -o out3.csv -c "+proc_ids+\
					" -g L2_DATA_READ_MISS_MEM_FILL:PMC0,DATA_CACHE_LINES_WRITTEN_BACK:PMC1 env "+env1+"./"+kernel_exec+" "+kernel_args

				ret = subprocess.call("ssh mic0 "+cmd,shell=True)
				if (ret != 0):
					print "Error"
					sys.exit(1)

				cmd = "KMP_AFFINITY="+affinity+" KMP_PLACE_THREADS="+str(cores)+"c,"+str(threads)+"t,00o"+\
					" likwid_mic/bin/likwid-perfctr -o out4.csv -c "+proc_ids+\
					" -g L2_DATA_WRITE_MISS_MEM_FILL:PMC0 env "+env1+"./"+kernel_exec+" "+kernel_args

				ret = subprocess.call("ssh mic0 "+cmd,shell=True)
				if (ret != 0):
					print "Error"
					sys.exit(1)


				cmd = "scp mic0:/home/echristof/*.csv ."
				if subprocess.call(cmd,shell=True) != 0:
					print "Error"
					sys.exit(1)

				#out_file = kernel.split('.')[0]+"_"+str((cntr1-1)*9+cntr2)+".csv"
				
				cmd = "cat out1.csv out2.csv out3.csv out4.csv > out.csv"
				if subprocess.call(cmd,shell=True) != 0:
					print "Error"
					sys.exit(1)

				parse_likwid.parse("out.csv","pwr.csv",conf,out_file)

				cmd = "rm pwr.csv out*"
				if subprocess.call(cmd,shell=True) != 0:
					print "Error"
					sys.exit(1)

				cntr2 += 1


# #main program
# if len(sys.argv) < 3:
# 	print "Please provide an application, its destination folder and its arguments if it needs."
# 	sys.exit(1)

# #take out the path
# path = sys.argv[1].split("/")
# kernel = path[len(path)-1]
# del path[len(path)-1]
# path = "/".join(path)

# kernel_exec = kernel.split(".")[0]+".xphi"
# if len(sys.argv) == 4:
# 	kernel_args = sys.argv[3]
# else:
# 	kernel_args = ""

# dst = sys.argv[2]

# if kernel.split(".")[1] == "c":
# 	CC="icc"
# elif kernel.split(".")[1] == "cpp":
# 	CC="icpc"
# else:
# 	print "Source code unknown"
# 	sys.exit(1)

kernel_exec = "nn"
args = ["filelist_64","10","42","24"]
CC = "icc"

out_file = "results/nn_filelist64_10.csv"
with open(out_file,"w+") as csvfile:
	headers = ['Configuration','grade1','CPI', 'BW/s((GB/sec)', 'GFLOPS/s', 'Vectorization', 'Power(W)', 'Temperature(C)','Time(sec)']
	writer = csv.writer(csvfile, delimiter = ',')
	writer.writerow(headers)

csvfile.close()

cntr1 = 1		
conf = []

compilation_flags = "-mmic -openmp -fno-alias -Wl,-rpath=/home/echristof/libraries "
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
						cntr1 += 1
				else:
					for f4 in streaming_stores2:
						for f5 in unroll:
							compiler()
							performer()
							cntr1 += 1
							



		