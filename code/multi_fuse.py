import numpy as np
import numpy as np
import os
import shutil
import multiprocessing
from skimage import io
from PIL import Image
from skimage import transform,data
import warnings
import cv2
warnings.filterwarnings('ignore')

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


def compute_mean_ioU(m,val_id,root1,root2,new_root):
    confusion_matrix = np.zeros((20, 20))
    a = m * 0.1
    b = 1 - a

    for i, im_name in enumerate(val_id):
        im_name = im_name.strip()
        gt_path = os.path.join('/home/liuwu1/notespace/dataset/LIP/val_labels/', im_name + '.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = np.asarray(gt, dtype=np.int32)
        
        logit1 = np.load(os.path.join(root1,im_name+'.npy'))
        logit2 = np.load(os.path.join(root2,im_name+'.npy'))
        logit = a*logit1 + b*logit2
        
        np.save(os.path.join(new_root, im_name+'.npy'),logit)
        
        pred = np.argmax(logit, axis=2)
        pred = np.asarray(pred, dtype=np.int32)
        
        ignore_index = gt != 255
        gt = gt[ignore_index]
        pred = pred[ignore_index]

        # print (logit1.shape, logit2.shape, logit.shape, gt.shape,pred.shape)

        confusion_matrix += get_confusion_matrix(gt, pred, 20)
        
        if i%1000==0:
            print ('***%.2f,%.2f***'%(a,b),i)

    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('===%.2f,%.2f==='%(a,b),'Pixel accuracy: %f; ' % pixel_accuracy,'Mean accuracy: %f; ' % mean_accuracy,'Mean IU: %f;' % mean_IoU)



def main():
    n_classes = 20
    #root1 = '/home/liuwu1/notespace/cvpr_workshop/parsing_pytorch/fuse_output/psp+deeplab_logits/'
    root1 = '/home/liuwu1/notespace/cvpr_workshop/parsing_pytorch/outputs/psp+deeplab_logits/'
    root2 = '/home/liuwu1/notespace/cvpr_workshop/parsing_pytorch/outputs/braid+hrnet_logits/'
    new_root = '../outputs/braid+hrnet+deeplab+psp_logits/'

    reader = open('val_id.txt')
    lines = reader.readlines()
    val_id=[]
    for line in lines:
        line = line.strip()
        val_id.append(line)
    compute_mean_ioU(5,val_id,root1,root2,new_root)
    print (aaa)
    pool = multiprocessing.Pool() 
    for m in range(11):
        pool.apply_async(compute_mean_ioU, (m,val_id,root1,root2,new_root))

    pool.close()
    pool.join()
if __name__ == "__main__":
    main()


