import argparse
import numpy as np
import torch
import time
import sys
sys.path.append('../../')  
from PIL import Image as PILImage
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
from networks.danet.danet import DANet
from dataset.datasets_origin import LIPDataSet
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU,write_results
from copy import deepcopy


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
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument('--cfg',default='cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml',
                        help='experiment configure file name',
                        #required=True,
                        type=str)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--snapshot_dir", type=str, default="",
                        help="")
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
    parser.add_argument("--multi-grid", action="store_true", default=False,
                        help="use multi grid dilation policy")
    parser.add_argument('--multi-dilation', nargs='+', type=int, default=None,
                        help="multi grid dilation list")
    return parser.parse_args()
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
def valid(model, valloader, input_size, num_samples, gpus):
    model.eval()
    time_list = []
    palette = get_lip_palette()  
    parsing_preds = np.zeros((num_samples, input_size[0], input_size[1]), dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, meta = batch
            #print (image.size())
            num_images = image.size(0)
            if index % 100 == 0:
                print('%d  processd' % (index * num_images))
            # if index ==100:
                # break
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
                parsing = outputs[0]
                parsing = interp(parsing)
                parsing = F.softmax(parsing,dim=1).data.cpu().numpy()
                parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC

                
                parsing = np.asarray(np.argmax(parsing, axis=3), dtype=np.uint8)
                parsing_preds[idx:idx + num_images, :, :] = parsing

                idx += num_images
                # for i in range(num_images):
                    # output_im = PILImage.fromarray(parsing[i,:,:]) 
                    # output_im.putpalette(palette)
                    # output_im.save('./outputs/%s.png'%name[i]+'.png')
            # break

    parsing_preds = parsing_preds[:num_samples, :, :]



    return parsing_preds, scales, centers, time_list

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    print (args)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)

    model = DANet(nclass=args.num_classes, backbone='resnet101' ,dilated=True, multi_grid=args.multi_grid,multi_dilation=args.multi_dilation)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    lip_dataset = LIPDataSet(args.data_dir, 'val', crop_size=input_size, transform=transform)
    num_samples = len(lip_dataset)

    valloader = data.DataLoader(lip_dataset, batch_size=args.batch_size * len(gpus),
                                shuffle=False, pin_memory=True)

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

    model.eval()
    model.cuda()

    parsing_preds, scales, centers,time_list= valid(model, valloader, input_size, num_samples, len(gpus))
    mIoU = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size)
    # write_results(parsing_preds, scales, centers, args.data_dir, 'val', args.save_dir, input_size=input_size)
    # write_logits(parsing_logits, scales, centers, args.data_dir, 'val', args.save_dir, input_size=input_size)
    
    

    print(mIoU)
    print('total time is ',sum(time_list))
    print('avg time is ',sum(time_list)/len(time_list))

if __name__ == '__main__':
    main()
