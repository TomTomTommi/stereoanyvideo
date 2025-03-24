#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

python train_stereoanyvideo.py --batch_size 1 \
 --spatial_scale -0.2 0.4 --image_size 256 512 --saturation_range 0 1.4 --num_steps 80000  \
 --ckpt_path logging/StereoAnyVideo_SF \
 --sample_len 5 --train_iters 10 --lr 0.0001 \
 --num_workers 8 --save_steps 3000 --train_datasets things monkaa driving
