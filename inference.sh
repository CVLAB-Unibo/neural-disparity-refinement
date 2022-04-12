#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

#args
backbone="vgg13"
algorithm="rSGM"

# path
ckpt="./checkpoints/sceneflow/net_latest"
results_path="./output/refined"
rgb="./sample/rgb_middlebury.png"
disparity="./sample/disp_middlebury.png"

# testing settings
max_disp=256
upsampling_factor=1
disp_scale=1
downsampling_factor=1

# extras
extras=""

python3 apps/inference.py --load_checkpoint_path $ckpt \
                        --backbone $backbone \
                        --results_path $results_path \
                        --upsampling_factor $upsampling_factor \
                        --results_path $results_path \
                        --max_disp $max_disp \
                        --disp_scale $disp_scale \
                        --downsampling_factor $downsampling_factor \
                        --rgb $rgb \
                        --disparity $disparity \
                        $extras
