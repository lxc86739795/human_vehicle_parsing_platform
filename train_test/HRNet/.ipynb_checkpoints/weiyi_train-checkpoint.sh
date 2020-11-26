#!/bin/bash
uname -a
#date
#env
date
export KMP_INIT_AT_FORK=FALSE

CS_PATH='/home/liuxinchen3/notespace/data/WeiyiAll/data'
LR=1e-2
WD=5e-4
BS=1 # 16 for 512*512, 4 for 1024*1024, 2 for 2048*1024, 1 for 2560*280)
# INPUT_SIZE='512,512'
# INPUT_SIZE='1024,1024'
INPUT_SIZE='2048,1024'
# INPUT_SIZE='2560,1280'
GPU_IDS='0,1,2,3'
RESTORE_FROM='/home/liuxinchen3/notespace/project/parsing_platform/LIP_models_trained/hrnet_w48-8ef0771d.pth'
SNAPSHOT_DIR='/home/liuxinchen3/notespace/project/parsing_platform/train_test/HRNet/models/hrnet_wy_1019_0000'
DATASET='train'
NUM_CLASSES=11
EPOCHS=100
START=0
SAVE_STEP=10

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

# python -u train_wy.py --data-dir ${CS_PATH} \
#       --restore-from ${RESTORE_FROM}\
#       --gpu ${GPU_IDS}\
#       --learning-rate ${LR}\
#       --weight-decay ${WD}\
#       --batch-size ${BS} \
#       --input-size ${INPUT_SIZE}\
#       --snapshot-dir ${SNAPSHOT_DIR}\
#       --dataset ${DATASET}\
#       --num-classes ${NUM_CLASSES} \
#       --epochs ${EPOCHS} \
#       --start-epoch ${START} \
#       --save_step ${SAVE_STEP}

nohup python -u train_wy.py --data-dir ${CS_PATH} \
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
       --start-epoch ${START} \
       --save_step ${SAVE_STEP} > log_train_hrnet_wy_${DATASET}_1019_0000.txt 2>&1 &
