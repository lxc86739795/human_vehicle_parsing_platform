import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing

LABELS_VP_COARSE = ['Background', 'Roof', 'Front-windshield', 'Face', 'Left-window', \
					'Left-body', 'Right-window', 'Right-body', 'Rear-windshield', 'Rear']

LABELS_VP_FINE = ['Background', 'Left-head-light', 'Left-fog-light', 'Right-head-light', 'Right-fog-light', \
				'Left-rear-light', 'Right-rear-light', 'Roof-light', 'Left-front-door', 'Right-front-door', \
				'Left-back-door', 'Right-back-door', 'Left-mirror', 'Right-mirror', 'Left-front-fender', \
				'Right-front-fender', 'Left-rear-fender', 'Right-rear-fender', 'Front-logo', 'Rear-logo', \
				'Hood', 'Grille', 'Roof', 'Rear-door', 'Front-plate', \
				'Rear-plate', 'Front-bumper', 'Rear-bumper', 'Front-windshield', 'Rear-windshield', \
				'Left-front-window', 'Right-front-window', 'Left-back-window', 'Right-back-window', 'Left-corner-window', \
				'Right-corner-window', 'Left-front-wheel', 'Right-front-wheel', 'Left-rear-wheel', 'Right-rear-wheel', \
				'Spare-tire', 'Roof-plate', 'Bus-left-body', 'Bus-right-body', 'Bus-left-window', \
				'Bus-right-window', 'Truck-left-face', 'Truck-right-face', 'Truck-left-sill', 'Truck-right-sill', \
				'Truck-container-connector', 'Container-front-side', 'Container-left-side', 'Container-right-side', 'Container-inside', \
				'Container-top-side', 'Container-back-side', 'Truck-left-midwheels', 'Truck-right-midwheels']

def get_vp_coarse_palette():
    palette = [0,0,0,
               127, 127, 255, 
               255, 255, 127, 
               212, 212, 212, 
               127, 255, 255, 
               255, 212, 127, 
               212, 127, 127, 
               127, 212, 255, 
               127, 255, 212, 
               212, 127, 212]
    return palette

def get_vp_fine_palette():
    palette = [0,0,0,
				95, 95, 95, 
				95, 95, 159, 
				95, 95, 223, 
				95, 95, 255, 
				95, 159, 95, 
				95, 159, 159, 
				95, 159, 223, 
				95, 159, 255, 
				95, 223, 95, 
				95, 223, 159, 
				95, 223, 223, 
				95, 223, 255, 
				95, 255, 95, 
				95, 255, 159, 
				95, 255, 223, 
				95, 255, 255, 
				159, 95, 95, 
				159, 95, 159, 
				159, 95, 223, 
				159, 95, 255, 
				159, 159, 95, 
				159, 159, 159, 
				159, 159, 223, 
				159, 159, 255, 
				159, 223, 95, 
				159, 223, 159, 
				159, 223, 223, 
				159, 223, 255, 
				159, 255, 95, 
				159, 255, 159, 
				159, 255, 223, 
				159, 255, 255, 
				223, 95, 95, 
				223, 95, 159, 
				223, 95, 223, 
				223, 95, 255, 
				223, 159, 95, 
				223, 159, 159, 
				223, 159, 223, 
				223, 159, 255, 
				223, 223, 95, 
				223, 223, 159, 
				223, 223, 223, 
				223, 223, 255, 
				223, 255, 95, 
				223, 255, 159, 
				223, 255, 223, 
				223, 255, 255, 
				255, 95, 95, 
				255, 95, 159, 
				255, 95, 223, 
				255, 95, 255, 
				255, 159, 95, 
				255, 159, 159, 
				255, 159, 223, 
				255, 159, 255, 
				255, 223, 95, 
				255, 223, 159]
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


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, list_path, input_size=[473, 473], dataset='coarse_val'):
    reader = open(list_path)
    val_id = reader.readlines()[0:len(preds)]

    confusion_matrix = np.zeros((num_classes, num_classes))
    dataset_name = dataset.split('_')[0]
    for i, im_name in enumerate(val_id):
        im_name = im_name.strip().split('.')[0]
        gt_path = os.path.join(datadir, dataset_name + '_annotation', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if type(gt) == None:
            print('Error in read file ', gt_path)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

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

    if 'coarse' in dataset:
        LABELS = LABELS_VP_COARSE
    elif 'fine' in dataset:
        LABELS = LABELS_VP_FINE
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


def write_results(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473], list_path=None):
    result_root = os.path.join(result_dir, dataset + '_result/')
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    vis_root = os.path.join(result_dir, dataset + '_vis/')
    if not os.path.exists(vis_root):
        os.makedirs(vis_root)
    if 'coarse' in dataset:
        palette = get_vp_coarse_palette()
    elif 'fine' in dataset:
        palette = get_vp_fine_palette()

    id_path = os.path.join(datadir, dataset + '_id.txt')
    reader = open(id_path)
    data_list = reader.readlines()[0:len(preds)]
    dataset_name = dataset.split('_')[0]
    for im_name, pred_out, s, c in zip(data_list, preds, scales, centers):
        im_name = im_name.strip().split('.')[0]
        image_path = os.path.join(datadir, dataset_name + '_image', im_name + '.jpg')
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        pred = transform_parsing(pred_out, c, s, w, h, input_size)

        save_path = os.path.join(result_root, im_name + '.png')
        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.save(save_path)

        save_path = os.path.join(vis_root, im_name + '.png')
        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(save_path)

