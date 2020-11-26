import os
import numpy as np
import random
import torch
import cv2
import json
from torch.utils import data
from utils.transforms import get_affine_transform


class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None, list_path='/home/liuwu1/notespace/dataset/LIP/LIP_CIHP_train_path_new.txt' ):

        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.transform = transform
        self.dataset = dataset

        print ('ListPath is',list_path)

        self.im_list = [i_id.strip() for i_id in open(list_path)]
        num_gpu = 4
        length = int(len(self.im_list)/num_gpu)*num_gpu
        self.im_list = self.im_list[0:length]
        print ('Len is:', len(self.im_list), ' GPU Num is:', num_gpu)
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index].strip()
        image_path, label_path = im_name.split(' ')
        im_path = os.path.join(self.root, image_path)
        parsing_anno_path = os.path.join(self.root, label_path)

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get person center and scale
        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            # Get pose annotation
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset == 'train' or self.dataset == 'trainval':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]

                    person_center[0] = im.shape[1] - person_center[0] - 1
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': image_path,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset != 'train':
            return input, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

 
            label_parsing = torch.from_numpy(label_parsing)

            return input, label_parsing, meta
