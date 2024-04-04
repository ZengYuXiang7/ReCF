#!/bin/bash

# 定义变量
run_name='Experiment'
rounds=1 epochs=100 device='cpu'
batch_size=256 learning_rate=0.001 decay=0.001
experiment=0 record=0 program_test=0 verbose=1
dimensions="32"
datasets="rt"
densities="0.10"
py_files="run_experiment"
models="GATCF"
#aggs="cat mean add"
aggs="cat mean add"

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
                    for agg in $aggs
                    do
                        python ./$py_file.py \
                          --logger $run_name \
                          --rounds $rounds \
                          --density $density \
                          --dataset $dataset \
                          --model $model \
                          --bs $batch_size \
                          --epochs $epochs \
                          --bs $batch_size \
                          --lr $learning_rate \
                          --decay $decay \
                          --program_test $program_test \
                          --dimension $dim \
                          --experiment $experiment \
                          --record $record \
                          --verbose $verbose \
                          --agg $agg
                    done
                done
            done
        done
    done
done