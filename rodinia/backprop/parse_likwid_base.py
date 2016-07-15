import csv
import collections
import sys

def get_output(output_file, stat_str):
	fp = open(output_file)	

	line = fp.readline().rstrip()
	while line:
		tokens = line.split(',')
		if stat_str in tokens:
			if stat_str == 'Runtime (RDTSC) [s]':
				#Take max for time, all 4 runs 
				ret = [float(tokens[2])]
				line = fp.readline().rstrip()
				times = 3
				while times > 0: 
					tokens = line.split(',')
					if stat_str in tokens:
						ret.append(float(tokens[2]))
						times -= 1
					line = fp.readline().rstrip()
			else:
				#take avg for every counter
				ret = float(tokens[2])

			break
		
		line = fp.readline().rstrip()
	
	fp.close()
	return ret

def get_output_pwr_tmp(output_file,power,temp):
	fp = open(output_file)	

	line = fp.readline()
	line = fp.readline().rstrip()
	while line:
		tokens = line.split(',')
		power.append(float(tokens[1].lstrip('"').rstrip('"')))
		temp.append(float(tokens[2].lstrip('"').rstrip('"')))

		line = fp.readline().rstrip()
	
	fp.close()
	return

def get_configuration(output_file):
	fp = open(output_file)
	line = fp.readline().rstrip()
	if line:
		tokens = line.split(',')
		ret = [tokens[2]]+[tokens[4]]
	fp.close()
	return ret


def parse(input_file1,input_file2,conf,output_file):
	# config = {}
	# config['KMP_AFFINITY'] = conf[6]
	# config['OMP_NUM_THREADS'] = conf[7]

	parameters = {'Time':'Runtime (RDTSC) [s]',
					'VPU_INSTRUCTIONS_EXECUTED':'VPU_INSTRUCTIONS_EXECUTED',
					'VPU_ELEMENTS_ACTIVE':'VPU_ELEMENTS_ACTIVE',
					'INSTRUCTIONS_EXECUTED':'INSTRUCTIONS_EXECUTED',
					'CPU_CLK_UNHALTED':'CPU_CLK_UNHALTED',
					'L2_DATA_READ_MISS_MEM_FILL':'L2_DATA_READ_MISS_MEM_FILL',
					'DATA_CACHE_LINES_WRITTEN_BACK':'DATA_CACHE_LINES_WRITTEN_BACK',
					'L2_DATA_WRITE_MISS_MEM_FILL':'L2_DATA_WRITE_MISS_MEM_FILL'} 
	metrics = {}

	for param,code in parameters.iteritems():
		metrics[param] = get_output(input_file1,code)

	power = []
	temp = []
	max_power = 0
	max_temp = 0
	
	get_output_pwr_tmp(input_file2,power,temp)
	max_temp = max(temp)
	max_power = max(power)

	# for p,m in metrics.iteritems():
	# 	print p,m

	CPI = metrics['CPU_CLK_UNHALTED'] / metrics['INSTRUCTIONS_EXECUTED']
	BW = (metrics['L2_DATA_WRITE_MISS_MEM_FILL'] + metrics['L2_DATA_READ_MISS_MEM_FILL'] + metrics['DATA_CACHE_LINES_WRITTEN_BACK']) * 64 * 1e-9 * 1
	Vector = metrics['VPU_ELEMENTS_ACTIVE'] / metrics['VPU_INSTRUCTIONS_EXECUTED'] 
	FLOPS = metrics['VPU_ELEMENTS_ACTIVE'] * 1e-9 * 1
	avg_time = sum(metrics['Time'])/len(metrics['Time'])

	grade1 = CPI + max_temp + max_temp
	headers = ['Configuration','grade1','CPI', 'BW/s(GB/sec)', 'GFLOPS/s', 'Vectorization', 'Power(W)', 'Temperature(C)','Time(sec)']
	results = [conf,round(grade1,4),round(CPI,4), round(BW/avg_time,4), round(FLOPS/avg_time,4), \
				round(Vector,4), max_power, max_temp, metrics['Time']]

	with open(output_file,'a') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(headers)
		writer.writerow(results)

	csvfile.close()	

	# results = {}
	# results['Configuration'] = conf
	# results['Time(sec)'] = metrics['Time']
	# results['CPI'] = CPI
	# results['BW/s(GB/sec)'] = BW / metrics['Time']
	# results['GFLOPS/s'] = FLOPS / metrics['Time']
	# results['Vector'] = Vector
	# results['Power(W)'] = max_power 
	# results['Temperature(C)'] = max_temp 

	# with open(output_file,'wb') as csvfile:
	# 	fieldnames = results.keys()
	# 	writer = csv.DictWriter(csvfile, delimiter = ',', 
	# 		fieldnames = fieldnames, quoting = csv.QUOTE_ALL)

	# 	writer.writeheader()
	# 	writer.writerow(results)

if __name__ == '__main__':
	print "Hello "+sys.argv[0]
	parse("out.csv","pwr.csv",'base 1 thread',sys.argv[1])