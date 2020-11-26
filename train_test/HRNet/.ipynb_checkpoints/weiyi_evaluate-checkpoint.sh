#!/bin/bash

CS_PATH='/home/liuxinchen3/notespace/data/WeiyiAll/data'
DATASET='val'
# DATASET='test_no_label'
NUM_CLASSES=11
BS=16
GPU_IDS='0,1,2,3'
# GPU_IDS='0'
INPUT_SIZE='2048,1024'
SNAPSHOT_FROM='/home/liuxinchen3/notespace/project/parsing_platform/train_test/HRNet/models/hrnet_wy_1019_0000/hrnet_epoch_70.pth'
OUTPUTS='./outputs/hrnet_wy_val_1019_0000_noflip'


if [[ ! -e ${OUTPUTS} ]]; then
    mkdir -p  ${OUTPUTS}
fi

# nohup python -u evaluate_wy.py --data-dir ${CS_PATH} \
#        --gpu ${GPU_IDS} \
#        --batch-size ${BS} \
#        --input-size ${INPUT_SIZE} \
#        --restore-from ${SNAPSHOT_FROM} \
#        --dataset ${DATASET} \
#        --num-classes ${NUM_CLASSES} \
#        --save-dir ${OUTPUTS} > ${OUTPUTS}/log_eval_hrnet_wy_${DATASET}_1012.txt 2>&1 &
       
python -u evaluate_wy.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE} \
       --restore-from ${SNAPSHOT_FROM} \
       --dataset ${DATASET} \
       --num-classes ${NUM_CLASSES} \
       --save-dir ${OUTPUTS}