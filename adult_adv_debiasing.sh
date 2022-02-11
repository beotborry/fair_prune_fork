for seed in 0 1 2 3 4
do

    python3 main.py --device 0 --dataset adult --sen-attr sex --lr 0.0005 --epoch 50 --iteration 50 --batch-size 128 --seed $seed --method reweighting --optimizer Adam --model mlp --reweighting-target-criterion eo --img-size 98 --weight-decay 0.0005 --date 20220210

done
