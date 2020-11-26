#!/bin/bash
uname -a
#date
#env
date

CS_PATH='/export/home/lxc/data/'
LR=4e-3
WD=5e-4
BS=6
GPU_IDS=2,3
RESTORE_FROM='../../LIP_models_trained/resnet101-imagenet.pth'
INPUT_SIZE='446,446'
SNAPSHOT_DIR='/export/home/lxc/parsing_platform/models/LIP_OCNet_0821/'
DATASET='train'
NUM_CLASSES=20
EPOCHS=100
START=0
LIST_PATH='/export/home/lxc/data/LIP/LIP_train_path.txt'
SAVE_STEP=2

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

# python -u train.py --data-dir ${CS_PATH} \
       # --random-mirror\
       # --random-scale\
       # --restore-from ${RESTORE_FROM}\
       # --gpu ${GPU_IDS}\
       # --learning-rate ${LR}\
       # --weight-decay ${WD}\
       # --batch-size ${BS} \
       # --input-size ${INPUT_SIZE}\
       # --snapshot-dir ${SNAPSHOT_DIR}\
       # --dataset ${DATASET}\
       # --num-classes ${NUM_CLASSES} \
       # --epochs ${EPOCHS} \
       # --start-epoch ${START}\
       # --list_path ${LIST_PATH}\
       # --save_step ${SAVE_STEP}

nohup python -u train.py --data-dir ${CS_PATH} \
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
       --save_step ${SAVE_STEP} > log_0821.txt 2>&1 &
