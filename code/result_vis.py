import os
import numpy as np
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
from PIL import Image as PILImage
warnings.filterwarnings('ignore')
def get_lip_palette():  
    palette = [ 0,0,0,
          128,0,0,
          255,0,0,
          0,85,0,
          170,0,51,
          255,85,0,
          0,0,85,
          0,119,221,
          85,85,0,
          0,85,85,
          85,51,0,
          52,86,128,
          0,128,0,
          0,0,255,
          51,170,221,
          0,255,255,
          85,255,170,
          170,255,85,
          255,255,0,
          255,170,0] 
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
root = '/home/liuwu1/notespace/cvpr_workshop/parsing_pytorch/train_test/HRNet_origin/outputs/CIHP_LIP_HRNet_ohem/val_logits'
npylist=os.listdir(root)
confusion_matrix = np.zeros((20, 20))
palette = get_lip_palette() 
for i, npy in enumerate(npylist):
    im_name = npy.split('.')[0]
    gt_path = os.path.join('/home/liuwu1/notespace/dataset/LIP/val_labels/', im_name + '.png')
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt = np.asarray(gt, dtype=np.int32)
    ignore_index = gt != 255
    gt = gt[ignore_index]
    
    logit = np.load(os.path.join(root,npy))
    pred = np.argmax(logit, axis=2)
    pred = np.asarray(pred, dtype=np.int32)
#     cv2.imwrite(os.path.join('./braid_hrnet_psp_deeplab_result/result/', im_name+'.png'), pred)
#     output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
#     output_im.putpalette(palette)
#     output_im.save(os.path.join('./braid_hrnet_psp_deeplab_result/vis/', im_name+'.png'))
    pred = pred[ignore_index]

    confusion_matrix += get_confusion_matrix(gt, pred, 20)

    if i%1000==0:
        print (i)
    

pos = confusion_matrix.sum(1)
res = confusion_matrix.sum(0)
tp = np.diag(confusion_matrix)

pixel_accuracy = (tp.sum() / pos.sum()) * 100
mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
IoU_array = (tp / np.maximum(1.0, pos + res - tp))
IoU_array = IoU_array * 100
mean_IoU = IoU_array.mean()
print('Pixel accuracy: %f; ' % pixel_accuracy,'Mean accuracy: %f; ' % mean_accuracy,'Mean IU: %f;' % mean_IoU)


