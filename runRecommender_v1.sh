#!/bin/bash

DIRECTORY="evaluation"

for fb in 0 1 
do
	for rank in $(seq 1 15)
	do
		OUT="./${DIRECTORY}/recommender_v1_${rank}_${fb}.out"
		>${OUT}
		for iter in $(seq 1 10)
		do
			python recommender_v1.py ${rank} ${fb} 0 >> ${OUT}
		done
	done
done
