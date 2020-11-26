#!/bin/bash

CS_PATH='/home/liuwu1/notespace/dataset/LIP/'
BS=32
GPU_IDS='1'
INPUT_SIZE='480,480'
SNAPSHOT_FROM='./models/CIHP_LIP_HRNet_Braid/LIP_epoch_9.pth'
DATASET='val'
NUM_CLASSES=20
OUTPUTS='./outputs/CIHP_LIP_HRNet_Braid/'
LIST_PATH='/home/liuwu1/notespace/dataset/LIP/val_id_3000.txt'
python eval_logits.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}\
       --list_path ${LIST_PATH}
