#!/bin/bash

CS_PATH='/home/liuwu1/notespace/dataset/LIP/'
BS=16
GPU_IDS='2'
INPUT_SIZE='480,480'
SNAPSHOT_FROM='./models/LIP_CIHP_psp/LIP_epoch_10.pth'
DATASET='val'
NUM_CLASSES=20
OUTPUTS='./outputs/LIP_CIHP_psp/'

python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}
