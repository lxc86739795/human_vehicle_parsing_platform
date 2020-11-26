#!/bin/bash

CS_PATH='/home/liuwu1/notespace/dataset/LIP/'
BS=32
GPU_IDS='1'
INPUT_SIZE='480,480'
SNAPSHOT_FROM='./models/LIP_CIHP_DenseASPP/LIP_epoch_29.pth'
DATASET='val'
NUM_CLASSES=20
OUTPUTS='./outputs/val_vis/'

python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}
