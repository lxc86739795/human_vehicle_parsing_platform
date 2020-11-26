import argparse
import numpy as np
import os
import time
import sys
sys.path.append('../../')  
from copy import deepcopy

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as transforms
torch.multiprocessing.set_start_method("spawn", force=True)

from networks.hrnet_v2_synbn import get_cls_net
from dataset.datasets import LIPDataSet, VPDataSet, WYDataSet
from utils.miou_VP import compute_mean_ioU, write_results
from utils.encoding import DataParallelModel, DataParallelCriterion

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


def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()
    time_list = []
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
	
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            num_images = image.size(0)
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            input = image.cuda()
            s_time = time.time()
            outputs = model(input)
            during_time = time.time() - s_time
            time_list.append(during_time)

            if gpus > 1:
                for output in outputs:
                    parsing = output
                    nums = len(parsing)
                    parsing = interp(parsing).data.cpu().numpy()
                    parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                    parsing = np.argmax(parsing, axis=3)
                    parsing_preds[idx:idx + nums, :, :] = np.asarray(parsing, dtype=np.uint8)
                    idx += nums
            else:
                parsing = outputs
                parsing = interp(parsing)
                parsing = F.softmax(parsing,dim=1).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC

                
                parsing = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                parsing_preds[idx:idx + num_images, :, :] = parsing

                idx += num_images

    parsing_preds = parsing_preds[:num_samples, :, :]

    return parsing_preds, scales, centers, time_list


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    update_config(config, args)
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)

    model = get_cls_net(config=config, num_classes=args.num_classes, is_train=False)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    print('-------Load Data', args.data_dir)
    if 'vehicle_parsing_dataset' in args.data_dir:
        parsing_dataset = VPDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    elif 'LIP' in args.data_dir:
        parsing_dataset = LIPDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    elif 'WeiyiAll' in args.data_dir:
        parsing_dataset = WYDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    
    num_samples = len(parsing_dataset)
    valloader = data.DataLoader(parsing_dataset, batch_size=args.batch_size * len(gpus), shuffle=False, pin_memory=True)

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
    parsing_preds, scales, centers, time_list = valid(model, valloader, input_size, num_samples, len(gpus))
    mIoU, no_test_class = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, list_path, input_size, dataset=args.dataset)
    print(mIoU)
    print('No test class : ', no_test_class)

    print('-------Saving Results', args.save_dir)
    write_results(parsing_preds, scales, centers, args.data_dir, args.dataset, args.save_dir, input_size=input_size)

    print('total time is ', sum(time_list))
    print('avg time is ', sum(time_list) / len(time_list))


if __name__ == '__main__':
    main()
