#!/bin/bash


# python3 main.py --device 0 --dataset adult --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed 777 --method adv_debiasing --optimizer Adam --model mlp --img-size 98 --weight-decay 0.0005 --date 20220211 --eta 0.0001 --adv-lambda 1 --target-criterion eopp


for seed in 0 1 2 3 4
do 
	for lambda in 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 20 30 50 100
	do
		for eta in 0.001 0.003 0.005 0.01
		do
			python3 main.py --device 0 --dataset adult --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method adv_debiasing --optimizer Adam --model mlp --img-size 98 --weight-decay 0.0005 --date 20220214 --eta $eta --adv-lambda $lambda --target-criterion eopp
			python3 main.py --device 0 --dataset adult --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method adv_debiasing --optimizer Adam --model mlp --img-size 98 --weight-decay 0.0005 --date 20220214 --eta $eta --adv-lambda $lambda --target-criterion eo
		done
	done
done
