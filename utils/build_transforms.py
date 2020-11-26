 # encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import numpy as np
import cv2
import torchvision.transforms as T

def build_transforms(args, is_train=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]
    res = []
    res_mask = []
    if 'train' in args.dataset:
        print('----build transform for training----')
        res.append(T.RandomHorizontalFlip(p=0.5))
        res.append(T.RandomVerticalFlip(p=0.5))
        if input_size[0] <= 1024:
            res.append(T.RandomCrop(input_size))
        
        transform = T.Compose(res)
        
        res_mask.append(T.RandomHorizontalFlip(p=0.5))
        res_mask.append(T.RandomVerticalFlip(p=0.5))
        if input_size[0] <= 1024:
            res_mask.append(T.RandomCrop(input_size))
        mask_transform = T.Compose(res_mask)

    elif 'val' in args.dataset:
        print('----build transform for testing----')
        transform = T.Compose([
#             T.ToTensor(),
#             normalize
        ])
        mask_transform = T.Compose([
#             T.ToTensor()
        ])
    elif 'test_no_label' in args.dataset:
        print('----build transform for test_no_label----')
        transform = T.Compose([
#             T.ToTensor(),
#             normalize
        ])
        mask_transform = None
    return (transform, mask_transform)
