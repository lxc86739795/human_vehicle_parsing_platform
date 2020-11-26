import argparse
import numpy as np
import os
import time
import sys
sys.path.append('../../')  
from copy import deepcopy

import cv2
import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as T
torch.multiprocessing.set_start_method("spawn", force=True)

from networks.hrnet_v2_synbn import get_cls_net
from dataset.datasets_wy import WYDataSet
from utils.miou_WY import compute_mean_ioU_wy, write_results_wy
from utils.encoding import DataParallelModel, DataParallelCriterion
from utils.collate_batch import fast_collate_fn, fast_collate_fn_mask
from utils.build_transforms import build_transforms
from utils.prefetcher import data_prefetcher, data_prefetcher_mask

from config import config
from config import update_config

DATA_DIRECTORY = '/ssd1/liuting14/Dataset/LIP/'
DATA_LIST_PATH = './dataset/list/lip/valList.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = (473,473)



def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="HRNET Network")
    parser.add_argument('--cfg',default='cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                        help='experiment configure file name',
                        #required=True,
                        type=str)
    parser.add_argument("--snapshot_dir", type=str, default="",
                        help="No use.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--save-dir", type=str, default='outputs',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    return parser.parse_args()


def valid(args, model, valloader, image_size, input_size, num_samples, gpus, use_flip_test=False):
    model.eval()
    time_list = []
    parsing_preds = np.zeros((num_samples, image_size[0], image_size[1]), dtype=np.uint8)
    is_rotated_all = np.zeros((num_samples), dtype=np.uint8)
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    
    y_block = image_size[0] // input_size[0] + 1
    x_block = image_size[1] // input_size[1] + 1
    
    print('Testing Image Number : ', num_samples)
    s_time = time.time()
    for y in range(y_block):
        for x in range(x_block):
            y_s = y*input_size[0]
            x_s = x*input_size[1]
            if y == y_block-1:
                y_s = image_size[0] - input_size[0]
            if x == x_block-1:
                x_s = image_size[1] - input_size[1]
            print('Processing block (', y, ',', x, '), total blocks : ', y_block*x_block)
            
            if 'test_no_label' not in args.dataset:
                val_prefetcher = data_prefetcher_mask(valloader)
            else:
                val_prefetcher = data_prefetcher(valloader)
                
            with torch.no_grad():
                idx = 0
                parsing_preds_block = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)
                batch = val_prefetcher.next()
                while batch[0] is not None:
                    if 'test_no_label' not in args.dataset: 
                        images, _, is_rotated = batch
                    else:
                        images, is_rotated = batch
                    num_images = images.size(0)
                    is_rotated_all[idx:idx + num_images] = is_rotated

                    block = images[:, :, y_s:y_s+input_size[0], x_s:x_s+input_size[1]]
                    input = block.cuda()
                    outputs = model(input)
                    if use_flip_test:
                        input_wf = block.data.cpu().numpy()
                        input_wf = input_wf[:, :, :, ::-1].copy()
                        input_wf = torch.from_numpy(input_wf)
                        outputs_wf = model(input_wf)
                        input_hf = block.data.cpu().numpy()
                        input_hf = input_hf[:, :, ::-1, :].copy()
                        input_hf = torch.from_numpy(input_hf)
                        outputs_hf = model(input_hf)
                        input_whf = block.data.cpu().numpy()
                        input_whf = input_whf[:, :, ::-1, ::-1].copy()
                        input_whf = torch.from_numpy(input_whf)
                        outputs_whf = model(input_whf)

                    if gpus > 1:
                        for i, output in enumerate(outputs):
                            parsing = output
                            nums = len(parsing)
                            parsing = interp(parsing).data.cpu().numpy()
                            if use_flip_test:
                                parsing_wf = interp(outputs_wf[i]).data.cpu().numpy()
                                parsing_wf = parsing_wf[:, :, :, ::-1].copy()
#                                 parsing = parsing+parsing_f
                                parsing_hf = interp(outputs_hf[i]).data.cpu().numpy()
                                parsing_hf = parsing_hf[:, :, ::-1, :].copy()
#                                 parsing = parsing+parsing_wf+parsing_hf
                                parsing_whf = interp(outputs_whf[i]).data.cpu().numpy()
                                parsing_whf = parsing_whf[:, :, ::-1, ::-1].copy()
                                parsing = parsing+parsing_wf+parsing_hf+parsing_whf
                            parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                            parsing = np.argmax(parsing, axis=3)
                            parsing_preds_block[idx:idx + nums, :, :] = np.asarray(parsing, dtype=np.uint8)
                            idx += nums
                    else:
#                         if use_flip_test:
#                             outputs = outputs+outputs_wf
                        parsing = outputs
                        parsing = interp(parsing)
                        parsing = F.softmax(parsing,dim=1).data.cpu().numpy()
                        parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC

                        parsing = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                        parsing_preds_block[idx:idx + num_images, :, :] = parsing

                        idx += num_images
                    print('Finished Image idx : ', idx)
                    batch = val_prefetcher.next()
                    
                    torch.cuda.empty_cache()

                parsing_preds[:, y_s:y_s+input_size[0], x_s:x_s+input_size[1]] = parsing_preds_block
    during_time = time.time() - s_time

    return parsing_preds, is_rotated_all, during_time


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    update_config(config, args)
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    image_size = (2788, 1400)

    model = get_cls_net(config=config, num_classes=args.num_classes, is_train=False)

    transform = build_transforms(args)

    print('-------Load Data : ', args.data_dir)
    parsing_dataset = WYDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    list_path = os.path.join(args.data_dir, parsing_dataset.list_file)
    
    num_samples = len(parsing_dataset)
    if 'test_no_label' not in args.dataset:
        valloader = data.DataLoader(parsing_dataset, batch_size=args.batch_size * len(gpus), shuffle=False, collate_fn=fast_collate_fn_mask, pin_memory=True)
    else:
        valloader = data.DataLoader(parsing_dataset, batch_size=args.batch_size * len(gpus), shuffle=False, collate_fn=fast_collate_fn, pin_memory=True)

    print('-------Load Weight', args.restore_from)
    restore_from = args.restore_from
    state_dict = model.state_dict().copy()
    state_dict_old = torch.load(restore_from)
    state_dict_old = state_dict_old['state_dict']

    for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
        if key != nkey:
            # remove the 'module.' in the 'key'
            state_dict[key[7:]] = deepcopy(state_dict_old[key])
        else:
            state_dict[key] = deepcopy(state_dict_old[key])

    model.load_state_dict(state_dict)
    model = DataParallelModel(model)

    model.eval()
    model.cuda()

    print('-------Start Evaluation...')
    parsing_preds, is_rotated, during_time = valid(args, model, valloader, image_size, input_size, num_samples, len(gpus))
    if 'test_no_label' not in args.dataset:
        mIoU, no_test_class = compute_mean_ioU_wy(parsing_preds, is_rotated, args.num_classes, args.data_dir, input_size, dataset=args.dataset, list_path = list_path)
        print(mIoU)
        print('No test class : ', no_test_class)

    print('-------Saving Results', args.save_dir)
    write_results_wy(parsing_preds, is_rotated, args.data_dir, args.dataset, args.save_dir, input_size=input_size, list_path = list_path)

    print('total time is ', during_time)
    print('avg time is ', during_time / num_samples)


if __name__ == '__main__':
    main()
