#!/bin/bash

CS_PATH='/export/home/lxc/data/'
BS=6
GPU_IDS='2,3'
INPUT_SIZE='446,446'
SNAPSHOT_FROM='/export/home/lxc/parsing_platform/models/LIP_OCNet/LIP_epoch_99.pth'
DATASET='val'
NUM_CLASSES=20
OUTPUTS='./outputs/LIP_OCNet/'
LIST_PATH='/export/home/lxc/data/LIP/val_id.txt'

if [[ ! -e ${OUTPUTS} ]]; then
    mkdir -p  ${OUTPUTS}
fi

nohup python -u evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS} >log_eval_0819.txt 2>&1 &
