#!/bin/bash
uname -a
#date
#env
date
export KMP_INIT_AT_FORK=FALSE

# # For VP_coarse
CS_PATH='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset'
LR=1e-2
WD=5e-4
BS=24
GPU_IDS=0,1,2,3
RESTORE_FROM='/home/liuxinchen3/notespace/project/parsing_platform/LIP_models_trained/hrnet_w48-8ef0771d.pth'
INPUT_SIZE='384,384'
SNAPSHOT_DIR='/home/liuxinchen3/notespace/project/parsing_platform/models/hrnet_vp_coarse_0229'
DATASET='coarse_train'
NUM_CLASSES=10
EPOCHS=150
START=0
LIST_PATH='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset/coarse_train_id.txt'
SAVE_STEP=10

# For VP_fine
# CS_PATH='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset'
# LR=1e-2
# WD=5e-4
# BS=24
# GPU_IDS=0,1,2,3
# RESTORE_FROM='/home/liuxinchen3/notespace/project/parsing_platform/LIP_models_trained/hrnet_w48-8ef0771d.pth'
# # RESTORE_FROM='/home/liuxinchen3/notespace/project/parsing_platform/models/hrnet_vp_coarse_0219/hrnet_epoch_150.pth'
# INPUT_SIZE='384,384'
# SNAPSHOT_DIR='/home/liuxinchen3/notespace/project/parsing_platform/models/hrnet_vp_fine_0228'
# DATASET='fine_train'
# NUM_CLASSES=59
# EPOCHS=150
# START=0
# LIST_PATH='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset/fine_train_id.txt'
# SAVE_STEP=10

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

# python -u train.py --data-dir ${CS_PATH} \
#       --random-mirror\
#       --random-scale\
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
#       --list_path ${LIST_PATH} \
#       --save_step ${SAVE_STEP}

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
       --start-epoch ${START} \
       --list_path ${LIST_PATH} \
       --save_step ${SAVE_STEP} > log_train_hrnet_VP_${DATASET}_0229.txt 2>&1 &
