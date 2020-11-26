#!/bin/bash
uname -a
#date
#env
date
CS_PATH='/home/liuwu1/notespace/dataset/CIHP/train'
LR=1e-2
WD=5e-4
BS=12
GPU_IDS=0,1,2,3
RESTORE_FROM='./models/CIHP_OCNet_GT/LIP_epoch_18.pth'
INPUT_SIZE='480,480'
SNAPSHOT_DIR='./models/CIHP_OCNet_GT/'
DATASET='train'
NUM_CLASSES=20
EPOCHS=20
START=19
LIST_PATH='/home/liuwu1/notespace/dataset/CIHP/CIHP_repeat_train_id_label_GT.txt'
SAVE_STEP=1

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python -u train.py --data-dir ${CS_PATH} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --epochs ${EPOCHS} \
       --start-epoch ${START}\
       --list_path ${LIST_PATH}\
       --save_step ${SAVE_STEP}
