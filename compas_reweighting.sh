#!/bin/bash

for seed in 0 1 2 3 4
do
    for eta in 10
    do
        python3 main.py --device 1 --dataset compas --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eopp --img-size 401 --eta $eta --weight-decay 0.0005
        python3 main.py --device 1 --dataset compas --sen-attr sex --lr 0.0005 --epoch 50 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eo --img-size 401 --eta $eta --weight-decay 0.0005

    done
done
