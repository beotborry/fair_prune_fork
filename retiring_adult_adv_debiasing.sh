#!/bin/bash

for seed in 0 1 2 3 4
do 
	for lambda in 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 20 30 50 100
	do
		for eta in 0.001 0.003 0.005 0.01
		do
			for k in 1 2 3 4 5 6 7 8 9 10
			do
			python3 main.py --device 2 --dataset retiring_adult --sen-attr race --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method adv_debiasing --optimizer Adam --model mlp --img-size 10 --weight-decay 0.0005 --date 20220315 --eta $eta --adv-lambda $lambda --target-criterion eopp --influence_removing 1 --k $k
			python3 main.py --device 2 --dataset retiring_adult --sen-attr race --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method adv_debiasing --optimizer Adam --model mlp --img-size 10 --weight-decay 0.0005 --date 20220315 --eta $eta --adv-lambda $lambda --target-criterion eo --influence_removing 1 --k $k
			done
		done
	done
done
