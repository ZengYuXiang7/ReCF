#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量
device='cpu'
densities="0.01"
experiment=1 record=0 program_test=0 verbose=1 classification=0

for density in $densities
do
		python train_model.py --config_path ./exper_config.py --exp_name MFConfig      --device $device
		python train_model.py --config_path ./exper_config.py --exp_name NeuCFConfig   --device $device
		python train_model.py --config_path ./exper_config.py --exp_name CSMFConfig    --device $device
		python train_model.py --config_path ./exper_config.py --exp_name GraphMFConfig --device $device
		python train_model.py --config_path ./exper_config.py --exp_name GATCFConfig   --device $device
done

