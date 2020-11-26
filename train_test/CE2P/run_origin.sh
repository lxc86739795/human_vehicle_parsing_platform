#!/bin/bash
uname -a
#date
#env
date

CS_PATH='/home/liuwu1/notespace/dataset/'
LR=1e-2
WD=5e-4
BS=8
GPU_IDS=3
RESTORE_FROM='../../models/CE2P_woEdge_145.pth'
INPUT_SIZE='384,384'
SNAPSHOT_DIR='../../models/LIP_CIHP_CE2P/'
DATASET='train'
NUM_CLASSES=20
EPOCHS=30
START=5

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python -u train_psp.py --data-dir ${CS_PATH} \
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
       --start-epoch ${START}
