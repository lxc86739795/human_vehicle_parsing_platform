import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing

LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']


def get_lip_palette():
    palette = [0, 0, 0,
               128, 0, 0,
               255, 0, 0,
               0, 85, 0,
               170, 0, 51,
               255, 85, 0,
               0, 0, 85,
               0, 119, 221,
               85, 85, 0,
               0, 85, 85,
               85, 51, 0,
               52, 86, 128,
               0, 128, 0,
               0, 0, 255,
               51, 170, 221,
               0, 255, 255,
               85, 255, 170,
               170, 255, 85,
               255, 255, 0,
               255, 170, 0]
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


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val', list_path=''):
    # val_file = os.path.join(datadir, 'annotations', dataset + '.json')
    # anno_file = open(val_file)
    # anno = json.load(anno_file)
    # anno = anno['root']
    # val_id = []
    # for i, a in enumerate(anno):
    # val_id.append(a['im_name'][:-4])
    reader = open(list_path)
    val_id = reader.readlines()[0:len(preds)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_id):
        im_name = im_name.strip()
        gt_path = os.path.join(datadir, dataset + '_labels', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
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


def compute_mean_ioU_file(preds_dir, num_classes, datadir, dataset='val'):
    list_path = os.path.join(datadir, dataset + '_id.txt')
    val_id = [i_id.strip() for i_id in open(list_path)]

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset + '_segmentations', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        pred_path = os.path.join(preds_dir, im_name + '.png')
        pred = np.asarray(PILImage.open(pred_path))

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    name_value = []

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        name_value.append((label, iou))

    name_value.append(('Pixel accuracy', pixel_accuracy))
    name_value.append(('Mean accuracy', mean_accuracy))
    name_value.append(('Mean IU', mean_IoU))
    name_value = OrderedDict(name_value)
    return name_value


def write_logits(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473], list_path=''):
    result_root = os.path.join(result_dir, dataset + '_logits/')
    if not os.path.exists(result_root):
        os.makedirs(result_root)
        print ('Make Dir: ', result_root)

    id_path = os.path.join(list_path)
    reader = open(id_path)
    data_list = reader.readlines()[0:len(preds)]
    count = 0
    for im_name, pred_out, s, c in zip(data_list, preds, scales, centers):
        if count % 100 == 0:
            print ('Have Save Logits %d' % count)
        im_name = im_name.strip()
        image_path = os.path.join(datadir, dataset + '_images', im_name + '.jpg')
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        pred = transform_parsing(pred_out, c, s, w, h, input_size)

        save_path = os.path.join(result_root, im_name + '.npy')
        np.save(save_path, pred)
        count = count + 1


def write_results(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473], list_path=''):
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
    palette = get_lip_palette()

    id_path = os.path.join(list_path)
    reader = open(id_path)
    data_list = reader.readlines()[0:len(preds)]
    count = 0

    for im_name, pred_out, s, c in zip(data_list, preds, scales, centers):
        if count % 100 == 0:
            print ('Have Save Result: %d' % count)
        im_name = im_name.strip()
        image_path = os.path.join(datadir, dataset + '_images', im_name + '.jpg')
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

        count = count + 1


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV NetworkEv")
    parser.add_argument("--pred-path", type=str, default='',
                        help="Path to predicted segmentation.")
    parser.add_argument("--gt-path", type=str, default='',
                        help="Path to the groundtruth dir.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    palette = get_palette(20)
    # im_path = '/ssd1/liuting14/Dataset/LIP/val_segmentations/100034_483681.png'
    # #compute_mean_ioU_file(args.pred_path, 20, args.gt_path, 'val')
    # im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    # print(im.shape)
    # test = np.asarray( PILImage.open(im_path))
    # print(test.shape)
    # if im.all()!=test.all():
    #     print('different')
    # output_im = PILImage.fromarray(np.zeros((100,100), dtype=np.uint8))
    # output_im.putpalette(palette)
    # output_im.save('test.png')
    pred_dir = '/ssd1/liuting14/exps/lip/snapshot_pose_parsing_sbn_adam/results/LIP_pose_epoch4/'
    num_classes = 20
    datadir = '/ssd1/liuting14/Dataset/LIP/'
    compute_mean_ioU_file(pred_dir, num_classes, datadir, dataset='val')
