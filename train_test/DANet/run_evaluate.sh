#!/bin/bash

CS_PATH='/home/liuwu1/notespace/dataset/LIP/'
BS=16
GPU_IDS='2'
INPUT_SIZE='480,480'
SNAPSHOT_FROM='./models/LIP_DANet/LIP_epoch_28.pth'
DATASET='val'
NUM_CLASSES=20
OUTPUTS='./outputs/CIHP_LIP_HRNet_ohem/'

python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}\
       --multi-grid \
       --multi-dilation 4 8 16