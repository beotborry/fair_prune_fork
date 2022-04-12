#!/bin/bash

for seed in 0 1 2 3 4
do
    for eta in 0.1 0.2 0.3 0.5 1 2 3 5 10 20 30
    do
        for lr in 0.0001 0.0003 0.0005
        do
            for k in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5
            do
                python3 main.py --device 0 --dataset adult --sen-attr sex --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eopp --img-size 98 --eta $eta --weight-decay 0.0005 --iteration 10 --date 20220401 --influence_removing 1 --k $k
                # python3 main.py --device 0 --dataset adult --sen-attr sex --lr $lr --epoch 5 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eo --img-size 98 --eta $eta --weight-decay 0.0005 --iteration 10 --date 20220315 --influence_removing 1 --k $k
            done
        done
    done
done
