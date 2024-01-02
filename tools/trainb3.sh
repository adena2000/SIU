#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG='/home/s06007/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/efficientvit_coco-wholebody_288_b1_large.py'
GPUS=3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_VISIBLE_DEVICES=3,6,7

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3} \



