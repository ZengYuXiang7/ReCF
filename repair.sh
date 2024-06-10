#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
run_name='Experiment'
rounds=5 epochs=500 patience=30 device='cpu'
batch_size=1024 learning_rate=0.001 decay=0.001
experiment=1 record=0 program_test=0 verbose=1 classification=0
dimensions="64"
datasets="rt"
densities="0.10"
#train_sizes="5 50 100 500 900"
#py_files="train_model"
py_files="repair"
models="GATCF"

for py_file in $py_files
do
		for dim in $dimensions
		do
				for dataset in $datasets
				do
						for model in $models
						do
								for density in $densities
								do
										python ./$py_file.py \
													--device $device \
													--logger $run_name \
													--rounds $rounds \
													--density $density \
													--dataset $dataset \
													--model $model \
													--bs $batch_size \
													--epochs $epochs \
													--patience $patience \
													--bs $batch_size \
													--lr $learning_rate \
													--decay $decay \
													--program_test $program_test \
													--dimension $dim \
													--experiment $experiment \
													--record $record \
													--verbose $verbose \
													--classification $classification \

								done
						done
				done
		done
done


