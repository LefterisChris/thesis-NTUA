#!/bin/bash

DIRECTORY="evaluation"

# for ts in 0.01 0.02 0.05 0.1
# do
# 	for iter in $(seq 1 10)
# 	do
# 		./UVdec.out 12 $ts 1
# 	done
# done

for iter in $(seq 1 10)
do
	./UVdec.out 6 0.001 1
	./UVdec.out 6 0.002 1
	./UVdec.out 6 0.005 1
	./UVdec.out 6 0.01 1
	./UVdec.out 6 0.02 1
	./UVdec.out 6 0.05 1
	./UVdec.out 6 0.1 1
done