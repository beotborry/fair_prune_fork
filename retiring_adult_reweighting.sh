#!/bin/bash

for seed in 0 1 2 3 4
do
    for eta in 1 2 3 4 5
    do
        python3 main.py --device 2 --dataset retiring_adult --sen-attr race --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eopp --img-size 10 --eta $eta
    done
done