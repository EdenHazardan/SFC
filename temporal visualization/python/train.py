import os
import sys
import time
import numpy as np
import random
import argparse
import ast
from tqdm import trange
import logging
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from PIL import Image, ImageOps, ImageFilter, ImageFile

from lib.dataset.viper_video_dataset import viper_video_dataset_accel_semantic_visual, IterLoader
from lib.dataset.cityscapes_video_dataset import cityscapes_video_dataset_accel_temporal
from lib.model.accel import Accel101_fusion_test, Accel101
from lib.metrics import runningScore_test

label=[
    "road",
    "sidewalk",
    "building",
    "fence",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "car",
    "truck",
    "bus",
    "motocycle",
    "bicycle"]

def get_arguments():
    parser = argparse.ArgumentParser(description="Train SCNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--local_rank", type=int, help="index the replica")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--weight_res101_1", type=str, help="path to resnet18 pretrained weight")
    parser.add_argument("--weight_res101_2", type=str, help="path to resnet101 pretrained weight")
    parser.add_argument("--weight_flownet", type=str, help="path to flownet pretrained weight")
    parser.add_argument("--numpy_transform", action="store_true", help="numpy or tensor normalize.")
    parser.add_argument("--multi_level", action="store_true", help="multi-level training.")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--train_num_workers", type=int)
    parser.add_argument("--test_num_workers", type=int)
    parser.add_argument("--train_iterations", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--work_dirs", type=str)

    return parser.parse_args()

# For visualization
label_colours = list(map(tuple, [
    [128,64,128],
    [244,35,232],
    [70,70,70],
    [190,153,153],
    [250,170,30],
    [220,220,0],
    [107,142,35],
    [152,251,152],
    [70,130,180],
    [220,20,60],
    [0,0,142],
    [0,0,70],
    [0,60,100],
    [0,0,230],
    # [119,11,32],
    [0,0,0],  # the color of ignored label
]))

IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)

NUM_CLASSES = 15

def decode_labels(mask, num_images=1, num_classes=NUM_CLASSES):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    h, w = mask.shape
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.new('RGB', (w, h))
    # img = Image.new('RGB', (w, h))
    pixels = img.load()
    for j_, j in enumerate(mask[:, :]):
        for k_, k in enumerate(j):
            if k < num_classes:
                pixels[k_, j_] = label_colours[k]
    outputs = np.array(img)
    return outputs

def train():
    args = get_arguments()


    test1_data = cityscapes_video_dataset_accel_temporal(split='val', numpy_transform=args.numpy_transform, distance=args.distance)
    test1_loader = DataLoader(
        test1_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        drop_last=False,
        pin_memory=True
    )

    net = Accel101_fusion_test(weight_res101_1=args.weight_res101_1, weight_res101_2=args.weight_res101_2, weight_flownet=args.weight_flownet, multi_level=args.multi_level).cuda()
    weight = torch.load('../save_results/SFC/best_for_Accel.pth')
    
    net.load_state_dict(weight, strict=True)

    print('begin validation')

    net.eval()

    save_city_path = '../SFC/temporal_semantic_visualization'

    # city
    test_loader_iter = iter(test1_loader)
    with torch.no_grad():
        i = 0
        for data in test_loader_iter:
            if i % 50 ==0 :
                print("test {}/{}".format(i,len(test1_loader)))
            im_seg_list, im_flow_list= data

            for j in range(29):
                if 1: 
                    img_s_list = torch.cat([im_seg_list[j],im_seg_list[j+1]], dim=2)
                    img_f_list = torch.cat([im_flow_list[j],im_flow_list[j+1]], dim=2)
                    pred_prop, pred_curr, pred_merge = net(img_s_list.cuda(), img_f_list.cuda())
                    out_merge = torch.argmax(pred_merge, dim=1)
                    out_merge = out_merge.squeeze().cpu().numpy()
                    # visualize
                    vis_merge = decode_labels(out_merge)
                    vis_merge = vis_merge[:, :, ::-1]
                    if not os.path.exists(os.path.join(save_city_path, '{}'.format(i))):
                        os.makedirs(os.path.join(save_city_path, '{}'.format(i)))
                    cv2.imwrite(os.path.join(save_city_path, '{}'.format(i), '{}.jpg'.format(j)), vis_merge)
            i += 1


if __name__ == '__main__':
    train()