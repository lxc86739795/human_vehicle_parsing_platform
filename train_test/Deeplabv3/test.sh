#!/bin/bash

CS_PATH='/export/home/zm/dataset/LIP/'
BS=32
GPU_IDS='0'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./models/snapshots1/LIP_epoch_145_54.47.pth'
DATASET='test'
NUM_CLASSES=20
OUTPUTS='./outputs/'

python test.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}
