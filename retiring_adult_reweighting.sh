#!/bin/bash

for seed in 0 1 2 3 4
do
    for eta in 0.1 0.2 0.3 0.5 1 2 3 5 10 20 30
    do
        for lr in 0.0001 0.0003 0.0005
        do
            for k in 1 2 3 4 5 6 7 8 9 10
            do
                python3 main.py --device 2 --dataset retiring_adult --sen-attr race --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eopp --img-size 10 --eta $eta --weight-decay 0.0005 --date 20220315
                python3 main.py --device 2 --dataset retiring_adult --sen-attr race --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eo --img-size 10 --eta $eta --weight-decay 0.0005 --date 20220315
            done
        done
    done
done
