#!/bin/bash

##For VP_coarse
CS_PATH='/export2/home/lxc/data/vehicle_parsing_dataset'
DATASET='coarse_val'
NUM_CLASSES=10
BS=48
GPU_IDS='0,1,2,3'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='/export2/home/lxc/project/parsing_platform/models/deeplabv3_vp_coarse_0228/deeplabv3_epoch_150.pth'
OUTPUTS='./outputs/deeplabv3_vp_coarse_val_0228'

# For VP_fine
# CS_PATH='/export2/home/lxc/data/vehicle_parsing_dataset'
# BS=48
# GPU_IDS='0,1,2,3'
# INPUT_SIZE='384,384'
# SNAPSHOT_FROM='/export2/home/lxc/project/parsing_platform/models/deeplabv3_vp_fine_0228/deeplabv3_epoch_150.pth'
# OUTPUTS='./outputs/deeplabv3_vp_fine_val_0228'
# DATASET='fine_val'
# NUM_CLASSES=59

if [[ ! -e ${OUTPUTS} ]]; then
    mkdir -p  ${OUTPUTS}
fi

nohup python -u evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE} \
       --restore-from ${SNAPSHOT_FROM} \
       --dataset ${DATASET} \
       --num-classes ${NUM_CLASSES} \
       --save-dir ${OUTPUTS} > ${OUTPUTS}/log_eval_deeplabv3_vp_${DATASET}_0228.txt 2>&1 &