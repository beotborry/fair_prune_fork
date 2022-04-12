#!/bin/bash
# for test
# python3 main.py --device 1 --dataset compas --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed 777 --method adv_debiasing --optimizer Adam --model mlp --img-size 401 --weight-decay 0.0005 --date 20220211 --eta 0.001 --adv-lambda 1 --target-criterion eopp


for seed in 0 1 2 3 4
do 
	for lambda in 0.001 0.003 0.1 0.03 0.1 0.3 1.0 3.0 10.0 20.0 30.0 50.0 100.0
	do
		for eta in 0.001 0.003 0.005 0.01
		do
			for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
			do
			python3 main.py --device 1 --dataset compas --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method adv_debiasing --optimizer Adam --model mlp --img-size 401 --weight-decay 0.0005 --date 20220401 --eta $eta --adv-lambda $lambda --target-criterion eopp --influence_removing 1 --k $k
			# python3 main.py --device 1 --dataset compas --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method adv_debiasing --optimizer Adam --model mlp --img-size 401 --weight-decay 0.0005 --date 20220315 --eta $eta --adv-lambda $lambda --target-criterion eo --influence_removing 1 --k $k
			done
		done
	done
done
