#!/bin/bash

for seed in 0 1 2 3 4
do
    for eta in 0.1 0.2 0.3 0.5 1 2 3 5 10 20 30
    do
        for lr in 0.001 0.0003 0.0005
        do
            # python3 main.py --device 1 --dataset adult --sen-attr sex --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eopp --img-size 98 --eta $eta --weight-decay 0.0005 --iteration 10 --date 20220211
            python3 main.py --device 0 --dataset adult --sen-attr sex --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eo --img-size 98 --eta $eta --weight-decay 0.0005 --iteration 10 --date 20220214
        done
    done
done
