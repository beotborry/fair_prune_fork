#!/bin/bash

for seed in 0 1 2 3 4
do
    for eta in 0.1 0.2 0.3 0.5 1 2 3 5 10 20 30
    do
        for lr in 0.0001 0.0003 0.0005
        do
            for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
            do
            python3 main.py --device 1 --dataset compas --sen-attr sex --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eopp --img-size 401 --eta $eta --weight-decay 0.0005 --iteration 10 --date 20220401 --influence_removing 1 --k $k
            # python3 main.py --device 1 --dataset compas --sen-attr sex --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eo --img-size 401 --eta $eta --weight-decay 0.0005 --iteration 10 --date 20220315 --influence_removing 1 --k $k
            done
        done
    done
done
