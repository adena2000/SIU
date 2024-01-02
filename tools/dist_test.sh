#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.

CONFIG='/home/s06007/mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_res50_8xb64-210e_coco-wholebody-256x192.py'
CHECKPOINT='/home/s06007/mmpose/checkpoints/res50_coco_wholebody_256x192-9e37ed88_20201004.pth'
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_VISIBLE_DEVICES=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
