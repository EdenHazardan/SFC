import os
import sys
import time
import numpy as np
import random
import argparse
import ast
from tqdm import trange
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

from lib.dataset.viper_video_dataset import viper_video_dataset_SFM, IterLoader
from lib.model.SFM import SFM_train

def EPE(input_flow, target_flow, gt=None, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if gt is not None:
        # invalid flow is defined with ignore_label -1
        mask = gt == -1
        EPE_map = EPE_map[~mask]

    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def get_arguments():
    parser = argparse.ArgumentParser(description="Train SFM")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--weight_flownet_global", type=str, help="path to flownet_global pretrained weight")
    parser.add_argument("--weight_SFM", type=str, help="path to SFM pretrained weight")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--train_num_workers", type=int)
    parser.add_argument("--test_num_workers", type=int)
    parser.add_argument("--train_iterations", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--work_dirs", type=str)

    return parser.parse_args()


def train():

    args = get_arguments()
    print(args)
    if not os.path.exists(os.path.join(args.work_dirs, args.exp_name)):
        os.makedirs(os.path.join(args.work_dirs, args.exp_name))
    tblogger = SummaryWriter(os.path.join(args.work_dirs, args.exp_name))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    fh = logging.FileHandler(os.path.join(args.work_dirs, args.exp_name, '{}.log'.format(rq)), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('random seed:{}'.format(random_seed))

    net = SFM_train(weight_flownet_global=args.weight_flownet_global, weight_SFM=args.weight_SFM).cuda()

    # Only train SFM
    params = []
    for p in net.SFM.parameters():
        params.append(p)
    optimizer = optim.Adam(params=params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0004)


    train_data = viper_video_dataset_SFM(split='train', distance=args.distance)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    train_loader = IterLoader(train_loader)

    test_data = viper_video_dataset_SFM(split='val', distance=args.distance)
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        drop_last=False,
        pin_memory=True
    )

    # init EPE
    best_EPE = 12345

    for step in trange(args.train_iterations):
        net.train()

        im_flow_list, gt0, gt, gt_down4x = next(train_loader)
        flow_label, flow_pred = net(im_flow_list.cuda(), gt0.cuda(), gt.cuda())
        flow_label = flow_label.detach()
        loss = EPE(flow_pred, flow_label, gt=gt_down4x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % args.log_interval == 0:
            print('iter:{}/{} loss:{:.6f}'.format(step + 1, args.train_iterations, loss.item()))
            logger.info('iter:{}/{} loss:{:.6f}'.format(step + 1, args.train_iterations, loss.item()))
            tblogger.add_scalar('loss', loss.item(), step + 1)

        if (step + 1) % args.val_interval == 0:
        # if 1:
            print('begin validation')
            logger.info('begin validation')

            net.eval()

            # validate
            total_EPE = 0
            test_loader_iter = iter(test_loader)
            with torch.no_grad():
                i = 0
                for data in test_loader_iter:
                    i += 1
                    if i % 50 ==0 :
                        print("test {}/{}".format(i,len(test_loader)))
                    im_flow_list, gt0, gt, _ = data
                    flow_label, flow_pred = net(im_flow_list.cuda(), gt0.cuda(), gt.cuda())
                    EPEloss = EPE(flow_pred, flow_label)
                    total_EPE += EPEloss
                realEPE = total_EPE / len(test_loader)
                    

            tblogger.add_scalar('realEPE', realEPE, step + 1)
            if realEPE < best_EPE:
                best_EPE = realEPE
                save_path = os.path.join(args.work_dirs, args.exp_name, 'SFM_source_well_trained.pth')
                torch.save(net.SFM.state_dict(), save_path)
            print('step:{} current EPE:{:.4f} best EPE:{:.4f}'.format(step + 1, realEPE, best_EPE))
            logger.info('step:{} current EPE:{:.4f} best EPE:{:.4f}'.format(step + 1, realEPE, best_EPE))


def adjust_lr(args, optimizer, itr):
    warmup_itr = 1000
    warmup_lr = 0.00005

    if itr < warmup_itr:
        now_lr = warmup_lr
    else:
        now_lr = args.lr

    for group in optimizer.param_groups:
        group['lr'] = now_lr
    return now_lr


if __name__ == '__main__':
    train()
