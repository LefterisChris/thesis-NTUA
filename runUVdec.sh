#!/bin/bash

DIRECTORY="evaluation"

for fb in 0
do
	for rank in $(seq 4 8)
	do
		for ts in 0.01 0.02 0.05 0.1 0.2 0.5
		do
			OUT="./${DIRECTORY}/UVdec_${rank}_${ts}_${fb}.out"
			>${OUT}
			for iter in $(seq 1 10)
			do
				./UVdec ${rank} ${ts} ${fb} >> ${OUT}
			done	
		done
	done
done

# for fb in 0
# do
# 	for rank in $(seq 1 14)
# 	do
# 		OUT="./${DIRECTORY}/UVdec_${rank}_${fb}.out"
# 		>${OUT}
# 		for iter in $(seq 1 10)
# 		do
# 			./UVdec ${rank} 0.1 ${fb} >> ${OUT}
# 		done
# 	done
# done
