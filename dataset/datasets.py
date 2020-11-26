import os
import numpy as np
import random
import glob
import torch
import cv2
import json
from torch.utils import data
from utils.transforms import get_affine_transform
from PIL import Image


class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):

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

        list_path = os.path.join('/home/liuwu1/notespace/dataset/LIP/LIP_CIHP_train_path_new.txt')

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


class VPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None, list_path=None):

        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.transform = transform
        self.dataset = dataset
        
        if 'other' in self.dataset:
            list_path = os.path.join(self.root, list_path)
        else:
            list_path = os.path.join(self.root, self.dataset + '_id.txt')

        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.im_list = self.im_list
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):   #keep h and w rate same
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
        if 'other' in self.dataset:
            im_name = self.im_list[index]
            im_path = os.path.join(self.root, im_name)
        else:
            im_name = self.im_list[index].split('.')[0]
            dataset_name = self.dataset.split('_')[0]

            im_path = os.path.join(self.root, dataset_name + '_image', im_name + '.jpg')
            parsing_anno_path = os.path.join(self.root, dataset_name + '_annotation', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if type(im) == None:
            print('Error in read file ', im_path)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get image center and scale
        image_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if 'test' not in self.dataset and 'other' not in self.dataset:
            # Get annotation
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if 'train' in self.dataset or 'trainval' in self.dataset :

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]

                    image_center[0] = im.shape[1] - image_center[0] - 1
                    if 'coarse' in self.dataset:
                        left_idx = [4, 5]
                        right_idx = [6, 7]
                    else:
                        left_idx = [1, 2, 5, 8, 10, 12, 14, 16, 30, 32, 34, 36, 38, 42, 44, 46, 48, 52, 57]
                        right_idx = [3, 4, 6, 9, 11, 13, 15, 17, 31, 33, 35, 37, 39, 43, 45, 47, 49, 53, 58]
                    for i in range(len(left_idx)):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(image_center, s, r, self.crop_size)
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
            'name': im_name,
            'center': image_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if 'train' not in self.dataset:
            return input, meta
        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))

            return input, label_parsing, meta
