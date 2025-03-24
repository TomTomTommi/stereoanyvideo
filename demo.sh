#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

python demo.py --ckpt ./checkpoints/StereoAnyVideo_MIX.pth --path ./demo_video/ --output_path ./demo_output/ --save_png