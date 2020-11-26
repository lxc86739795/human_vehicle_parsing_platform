#!/bin/bash

CS_PATH='/export/home/zm/dataset/LIP/val_images'
BS=32
GPU_IDS='0'
INPUT_SIZE='480,480'
SNAPSHOT_FROM='./models/LIP_epoch_9.pth'
DATASET='test'
LIST_PATH='./list/test_path.txt'
NUM_CLASSES=20
OUTPUTS='./outputs/'

python test.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}\
       --list_path ${LIST_PATH}
