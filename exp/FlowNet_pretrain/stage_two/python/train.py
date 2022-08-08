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
# import torch.distributed as dist
from torch.utils.data import DataLoader
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

from tensorboardX import SummaryWriter

from lib.dataset.viper_video_dataset import viper_video_dataset, IterLoader
from lib.dataset.cityscapes_video_dataset import cityscapes_video_dataset, IterLoader
from lib.model.accel import Accel101
from lib.metrics import runningScore


def get_arguments():
    parser = argparse.ArgumentParser(description="Train src-only stage two to obtain well pretrained FlowNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--weight_update", type=str, help="path to resnet18 pretrained weight")
    parser.add_argument("--weight_reference", type=str, help="path to resnet101 pretrained weight")
    parser.add_argument("--weight_flownet", type=str, help="path to flownet pretrained weight")
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

    net = Accel101(weight_update=args.weight_update, weight_reference=args.weight_reference, weight_flownet=args.weight_flownet, multi_level=args.multi_level).cuda()

    params = []
    for p in net.merge.parameters():
        params.append(p)
    for p in net.flownet.parameters():
        params.append(p)
    if args.multi_level:
        for p in net.net_update.layer5.parameters():
            params.append(p)
        for p in net.net_ref.layer5.parameters():
            params.append(p)
    for p in net.net_update.layer6.parameters():
        params.append(p)
    for p in net.net_ref.layer6.parameters():
        params.append(p)
    optimizer = optim.SGD(params=params, lr=args.lr, weight_decay=0.0005, momentum=0.9)

    # copy ADVENT
    train_data = viper_video_dataset(split='train', distance=args.distance)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    train_loader = IterLoader(train_loader)

    test1_data = cityscapes_video_dataset(split='val', distance=args.distance)
    test1_loader = DataLoader(
        test1_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        drop_last=False,
        pin_memory=True
    )

    miou_cal = runningScore(n_classes=15)
    best_city_miou = 0.0


    for step in trange(args.train_iterations):
        net.train()

        im_seg_list, im_flow_list, gt = next(train_loader)
        loss = net(im_seg_list.cuda(), im_flow_list.cuda(), gt.cuda())
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr = adjust_lr(args, optimizer, step)

        if (step + 1) % args.log_interval == 0:
            print('iter:{}/{} lr:{:.6f} loss:{:.6f}'.format(step + 1, args.train_iterations, lr, loss.item()))
            logger.info('iter:{}/{} lr:{:.6f} loss:{:.6f}'.format(step + 1, args.train_iterations, lr, loss.item()))
            tblogger.add_scalar('lr', lr, step + 1)
            tblogger.add_scalar('loss', loss.item(), step + 1)

        if (step + 1) % args.val_interval == 0:
        # if 1:
            print('begin validation')
            logger.info('begin validation')

            net.eval()

            # test city
            test_loader_iter = iter(test1_loader)
            with torch.no_grad():
                i = 0
                for data in test_loader_iter:
                    i += 1
                    if i % 20 ==0 :
                        print("test {}/{}".format(i,len(test1_loader)))
                    im_seg_list, im_flow_list, gt = data
                    pred = net(im_seg_list.cuda(), im_flow_list.cuda())
                    out = torch.argmax(pred, dim=1)
                    out = out.squeeze().cpu().numpy()
                    gt = gt.squeeze().cpu().numpy()
                    miou_cal.update(gt, out)
                miou = miou_cal.get_scores()
                miou_cal.reset()

            tblogger.add_scalar('miou_city', miou, step + 1)
            # mIoU can represent the quality of optical flow. The optical flow is required to propogate t-1 feature properly to obtain accurate feature fusion
            if miou > best_city_miou:
                best_city_miou = miou
                save_path = os.path.join(args.work_dirs, args.exp_name, 'flownet_source_well_trained.pth')
                torch.save(net.flownet.state_dict(), save_path)
            print('step:{} current city miou:{:.4f} best city miou:{:.4f}'.format(step + 1, miou, best_city_miou))
            logger.info('step:{} current city miou:{:.4f} best city miou:{:.4f}'.format(step + 1, miou, best_city_miou))


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
