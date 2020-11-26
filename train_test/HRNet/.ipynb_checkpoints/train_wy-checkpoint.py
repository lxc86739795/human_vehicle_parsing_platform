import argparse

import torch
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import numpy as np
import torch.optim as optim
import torchvision.utils as vutils
import time
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import sys
sys.path.append('../../')  

from networks.hrnet_v2_synbn import get_cls_net
from dataset.datasets_wy import WYDataSet
import torchvision.transforms as T
import timeit
from tensorboardX import SummaryWriter
from utils.utils import decode_parsing, inv_preprocess

from utils.criterion import CriterionAll
# from utils.loss import OhemCrossEntropy2d
# from utils.criterion2 import CriterionDSN
from utils.lovasz_losses import LovaszSoftmax, LovaszSoftmaxDSN

from utils.encoding import DataParallelModel, DataParallelCriterion 
from utils.miou import compute_mean_ioU
from utils.collate_batch import fast_collate_fn_mask
from utils.build_transforms import build_transforms
from utils.prefetcher import data_prefetcher_mask

from config import config
from config import update_config
from PIL import Image as PILImage


start = timeit.default_timer()
 
BATCH_SIZE = 8
DATA_DIRECTORY = 'WeiyiAll'
IGNORE_LABEL = 255
INPUT_SIZE = '512,512'
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
NUM_CLASSES = 11
POWER = 0.9
RANDOM_SEED = 618
RESTORE_FROM = './dataset/MS_DeepLab_resnet_pretrained_init.pth'
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
 
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--dataset", type=str, default='train', choices=['train', 'test', 'test_no_label'],
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--start-iters", type=int, default=0,
                        help="Number of classes to predict (including background).") 
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--gpu", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--list_path", type=str, default='None',
                        help="choose gpu device.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--save_step", type=int, default=0,
                        help="choose the number of recurrence.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="choose the number of recurrence.")
    parser.add_argument("--loss", type=str, default='softmax',
                        help="")
    return parser.parse_args()


args = get_arguments()
update_config(config, args)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, total_iters):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, total_iters, args.power)
    optimizer.param_groups[0]['lr'] = lr
    # for i in range(1,len( optimizer.param_groups)):
        # optimizer.param_groups[i]['lr'] = lr
    return lr


def main():
    """Create the model and start the training."""
    print (args)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    writer = SummaryWriter(args.snapshot_dir)
    gpus = [int(i) for i in args.gpu.split(',')]
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    # cudnn related setting
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


    deeplab = get_cls_net(config=config, num_classes=args.num_classes, is_train=True)
    
    print('-------Load Weight', args.restore_from)
    saved_state_dict = torch.load(args.restore_from)
	
    if args.start_epoch > 0:
        model = DataParallelModel(deeplab)
        model.load_state_dict(saved_state_dict['state_dict'])
    else:
        new_params = deeplab.state_dict().copy()
        state_dict_pretrain = saved_state_dict
        for state_name in state_dict_pretrain:
            if state_name in new_params:
                new_params[state_name] = state_dict_pretrain[state_name]
            else:
                print ('NOT LOAD', state_name)
        deeplab.load_state_dict(new_params)
        model = DataParallelModel(deeplab)
    print('-------Load Weight Finish', args.restore_from)
    
    model.cuda()

    criterion0 = CriterionAll(loss_type='ohem')
    criterion0 = DataParallelCriterion(criterion0)
    criterion0.cuda()
    
    criterion1 = LovaszSoftmax(input_size=input_size)
    criterion1 = DataParallelCriterion(criterion1)
    criterion1.cuda()

    transform = build_transforms(args)

    print("-------Loading data...")
    parsing_dataset = WYDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    print("Data dir : ", args.data_dir)
    print("Dataset : ", args.dataset, "Sample Number: ", parsing_dataset.number_samples)
    trainloader = data.DataLoader(parsing_dataset, 
                                  batch_size=args.batch_size * len(gpus), shuffle=True, num_workers=8,
                                  collate_fn=fast_collate_fn_mask, pin_memory=True)

    
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    if args.start_epoch > 0:
        optimizer.load_state_dict(saved_state_dict['optimizer'])
        print ('========Load Optimizer',args.restore_from)
    

    total_iters = args.epochs * len(trainloader)
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        tng_prefetcher = data_prefetcher_mask(trainloader)
        batch = tng_prefetcher.next()
        n_batch = 0
        while batch[0] is not None:
#         for i_iter, batch in enumerate(trainloader):
            i_iter = n_batch + len(trainloader) * epoch
            lr = adjust_learning_rate(optimizer, i_iter, total_iters)

            images, labels, _ = batch
            labels = labels.squeeze(1)
            labels = labels.long().cuda(non_blocking=True)
            preds = model(images)

            loss0 = criterion0(preds, labels)
            loss1 = criterion1(preds, labels)
            loss=loss0+loss1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch = tng_prefetcher.next()
            n_batch += 1
            

            if i_iter % 1 == 0:
                writer.add_scalar('learning_rate', lr, i_iter)
                writer.add_scalar('loss', loss.data.cpu().numpy(), i_iter)
                writer.add_scalar('loss0', loss0.data.cpu().numpy(), i_iter)
                writer.add_scalar('loss1', loss1.data.cpu().numpy(), i_iter)

            print(f'epoch = {epoch}, iter = {i_iter}/{total_iters}, lr={lr:.6f}, \
                  loss = {loss.data.cpu().numpy():.6f}, \
                  loss0 = {loss0.data.cpu().numpy():.6f}, \
                  loss1 = {loss1.data.cpu().numpy():.6f}')
        
        if (epoch+1) % args.save_step == 0 or epoch == args.epochs:
            time.sleep(10)
            print("-------Saving checkpoint...")
            save_checkpoint(model, epoch, optimizer)

    time.sleep(10)
    save_checkpoint(model, epoch, optimizer)
    end = timeit.default_timer()
    print(end - start, 'seconds')


def save_checkpoint(model, epoch, optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath = osp.join(args.snapshot_dir, 'hrnet_epoch_' + str(epoch+1) + '.pth')
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
