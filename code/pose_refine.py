# -*- coding: UTF-8 -*-
from __future__ import division
import math
import random
import scipy.misc
import numpy as np
from scipy.stats import multivariate_normal
import scipy.io as sio 
import csv
import cv2
import os
#1right Ankle,2right Knee,3right Hip,4leftHip,5leftKnee,6leftAnkle,7Pelvis,8Spine,9Neck,10Head,11right Wrist,12right Elbow,13right Shoulder,14left Shoulder,15left Elbow,16left Wrist 
#1右脚踝,2右膝盖,3右臀围,4左臀围,5左膝,6左脚踝,7骨盆,8脊柱,9颈部,10头部,11右手腕,12右肘,13右肩,14左肩,15左肘,16左手腕
csv_file = '/export/home/zm/dataset/LIP/TrainVal_pose_annotations/lip_val_set.csv'
root = '/export/home/zm/dataset/LIP/train_images/'
green = (0, 255, 0)
with open(csv_file, "r") as input_file:
    count = 0
    for row in csv.reader(input_file):
        count = count + 1

        img_id = row.pop(0)[:-4]
        print (img_id)

        pred_path = '/export/home/zm/test/cvpr_workshop/parsing_pytorch/train_test/HRNet/outputs/CIHP_LIP_HRNetv2_bn29/val_result/{}.png'.format(img_id)
        image_path = '/export/home/zm/dataset/LIP/val_images/{}.jpg'.format(img_id)
        pred=  cv2.imread(pred_path,0)
        img = scipy.misc.imread(image_path).astype(np.float)
        rows = img.shape[0]
        cols = img.shape[1]
        keys = np.zeros((16,2))#rows:行; cols:列
        keys = keys.astype(np.int)
        for idx, point in enumerate(row):
            if 'nan' in point:
                point = 0
            if idx % 3 == 0:
                c_ = int(point)
                c_ = min(c_, cols-1)
                c_ = max(c_, 0)
            elif idx % 3 == 1 :
                r_ = int(point)
                r_ = min(r_, rows-1)
                r_ = max(r_, 0)
                keys[int(idx / 3),0] = r_
                keys[int(idx / 3),1] = c_

        image = cv2.imread(image_path)
        xr1,yr1,xr2,yr2 = keys[1,1], keys[1,0], keys[0,1], keys[0,0]
        xl1,yl1,xl2,yl2 = keys[1,1], keys[1,0], keys[0,1], keys[0,0]

        if x1>x2:
            xrmin,xrmax = xr2-3, xr1+3
        else:
            xrmin,xrmax = xr1-3, xr2+3
        if y1>y2:
            yrmin,yrmax = yr2-3, yr1+3
        else:
            yrmin,yrmax = yr1-3, yr2+3
        
        pred = cv2.imread(pred_path,0)
        pred1 = np.where(pred[xrmin:xrmax,yrmin:yrmax],16,17)
        pred[xmin:xmax,ymin:ymax] = pred
        x = image.copy()
        x = x.astype(np.uint8)
        cv2.rectangle( x, (x1,y1), (x2,y2),  green,3)
        pred.imwrite('./outputs/%d.png'%count,x)
        # print (image_draw.shape)
        if count>2:
            break



