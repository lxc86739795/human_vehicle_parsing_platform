import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing

LABELS = [
    'BG', 
    'huashang',
    'heidian',
    'baidian',
    'bengbian',
    'yise',
    'juchi',
    'cashang',
    'yashang',
    'maosi',
    'zangwu']


def get_wy_palette():
    palette = [
    0,0,0,
    127, 127, 0,
    127, 127, 255,
    255, 255, 127,
    212, 212, 212,
    127, 255, 255,
    255, 212, 127,
    212, 127, 127,
    127, 212, 255,
    127, 255, 212,
    255, 255, 255
    ]

    return palette


def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


def compute_mean_ioU_wy(preds, is_rotated, num_classes, datadir, input_size=[473, 473], dataset='val', list_path=''):
    reader = open(list_path)
    val_list = reader.readlines()[:len(preds)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_list):
        im_name = im_name.strip()
        gt_path = os.path.join(datadir, 'annotations_all', im_name.replace('jpg', 'png'))
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if type(gt) == None:
            print('Error in read file ', gt_path)
        h, w = gt.shape
        pred = preds[i]

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)
        if is_rotated[i] == 1:
            pred = pred.transpose(1,0)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    has_test = res > 1
    
    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos))[has_test].mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
#     mean_IoU = IoU_array.mean()
    mean_IoU = IoU_array[has_test].mean() # ignore classes having no test samples
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []
    no_test_name = []

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        if has_test[i]:
            name_value.append((label, iou))
        else:
            no_test_name.append(label)

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value, no_test_name


def write_results_wy(preds, is_rotated, datadir, dataset, result_dir, input_size=[473, 473], list_path=''):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print ('Make Dir: ', result_dir)
    result_root = os.path.join(result_dir, dataset + '_result/')
    if not os.path.exists(result_root):
        os.makedirs(result_root)
        print ('Make Dir: ', result_root)
    vis_root = os.path.join(result_dir, dataset + '_vis/')
    if not os.path.exists(vis_root):
        os.makedirs(vis_root)
        print ('Make Dir: ', vis_root)
    palette = get_wy_palette()

    id_path = os.path.join(list_path)
    reader = open(id_path)
    data_list = reader.readlines()
    count = 0

    for im_name, pred, r in zip(data_list, preds, is_rotated):
        if count % 64 == 0:
            print ('Have Saved Result: %d' % count)
        im_name = im_name.strip()
        
        if r == 1:
            pred = pred.transpose(1,0)

        save_path = os.path.join(result_root, im_name.replace('jpg', 'png'))
        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.save(save_path)

        save_path = os.path.join(vis_root, im_name.replace('jpg', 'png'))
        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(save_path)

        count = count + 1
