#!/usr/bin/env python

import csv
import collections
import sys


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

power = []
temp = []

avg_power = 0
avg_temp = 0

for n in [1,2,3,4]:
	get_output_pwr_tmp("pwr"+str(n)+".csv",power,temp)

	avg_power += (sum(power)/len(power))/4
	avg_temp += (sum(temp)/len(temp))/4


print avg_power,avg_temp


