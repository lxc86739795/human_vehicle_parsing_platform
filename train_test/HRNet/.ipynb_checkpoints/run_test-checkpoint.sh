#!/bin/bash

# For VP_coarse
CS_PATH='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset/image_for_pseudo_label'
DATASET='coarse_other'
NUM_CLASSES=20
BS=48
GPU_IDS='0,1,2,3'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='/home/liuxinchen3/notespace/project/parsing_platform/models/hrnet_vp_coarse_0219/hrnet_epoch_150.pth'
OUTPUTS='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset/image_by_pseudo_label'
LIST_PATH='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset/no_label_all.txt'

# # For VP_fine
# CS_PATH='/home/liuxinchen3/notespace/data/vehicle_parsing_dataset'
# BS=64
# GPU_IDS='0,1,2,3'
# INPUT_SIZE='384,384'
# SNAPSHOT_FROM='/home/liuxinchen3/notespace/project/parsing_platform/models/hrnet_vp_fine_0220/hrnet_epoch_150.pth'
# OUTPUTS='./outputs/hrnet_vp_fine_val_0220'
# DATASET='fine_val'
# NUM_CLASSES=59

if [[ ! -e ${OUTPUTS} ]]; then
    mkdir -p  ${OUTPUTS}
fi

# nohup python -u test.py --data-dir ${CS_PATH} \
       # --gpu ${GPU_IDS} \
       # --batch-size ${BS} \
       # --input-size ${INPUT_SIZE} \
       # --restore-from ${SNAPSHOT_FROM} \
       # --dataset ${DATASET} \
       # --num-classes ${NUM_CLASSES} \
       # --save-dir ${OUTPUTS} > ./outputs/hrnet_vp_${DATASET}_0220/log_eval_hrnet_vp_${DATASET}_0225.txt 2>&1 &
	   
	   
python -u test.py --data-dir ${CS_PATH} \
	   --list_path ${LIST_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE} \
       --restore-from ${SNAPSHOT_FROM} \
       --dataset ${DATASET} \
       --num-classes ${NUM_CLASSES} \
       --save-dir ${OUTPUTS}