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

from lib.dataset.viper_video_dataset import viper_video_dataset, IterLoader
from lib.dataset.cityscapes_video_dataset import cityscapes_video_dataset, IterLoader
from lib.model.one_stage_img_model import src_only_one_stage_model
from lib.metrics import runningScore


ignore_index = -1
label_name=["road", "sidewalk", "building", "fence", "light", "sign", "vegetation", "terrain", "sky", "person", "car", "truck", "bus", "motocycle", "bicycle"]
NUM_CLASS = 15

def get_arguments():
    parser = argparse.ArgumentParser(description="Train scr-only one stage model")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")

    ###### training setting ######
    parser.add_argument("--weight_deeplabv2_res101", type=str, help="path to deeplabv2_res101 pretrained weight")
    parser.add_argument("--model_name", type=str, help="one stage model: update or reference?")
    parser.add_argument("--multi_level", action="store_true", help="multi-level training, following DAISS, e.g., ADVENT, PixMatch")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument("--source_batch_size", type=int)
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

    net = src_only_one_stage_model(weight_deeplabv2_res101=args.weight_deeplabv2_res101).cuda()

    params = net.segnet.optim_parameters(lr=args.lr)
    optimizer = optim.SGD(params=params, lr=args.lr, weight_decay=0.0005, momentum=0.9)

    # Source and target dataset (copy ADVENT)
    source_data = viper_video_dataset(split='train', distance=args.distance)
    source_loader = DataLoader(
        source_data,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    source_loader = IterLoader(source_loader)

    test_data = cityscapes_video_dataset(split='val', distance=args.distance)
    test_loader = DataLoader(
        test_data,
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

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        lr = poly_lr_scheduler(args, optimizer, iter=step)

        # train on source
        im_seg_list, im_flow_list, gt = next(source_loader)
        loss = net(im_seg_list.cuda(), im_flow_list.cuda(), gt=gt.cuda())
        source_loss = loss.mean()
        # print("source_loss = ",source_loss.item())
        source_loss.backward()

        # total_loss.backward()
        optimizer.step()

        if (step + 1) % args.log_interval == 0:
            print('iter:{}/{} lr:{:.6f} source_loss:{:.6f}'.format(step + 1, args.train_iterations, lr, source_loss.item()))
            logger.info('iter:{}/{} lr:{:.6f} source_loss:{:.6f}'.format(step + 1, args.train_iterations, lr, source_loss.item()))
            tblogger.add_scalar('lr', lr, step + 1)
            tblogger.add_scalar('source_loss', source_loss.item(), step + 1)


        if (step + 1) % args.val_interval == 0:
        # if 1:
            print('begin validation')
            logger.info('begin validation')

            net.eval()

            # test city
            test_loader_iter = iter(test_loader)
            with torch.no_grad():
                i = 0
                for data in test_loader_iter:
                    i += 1
                    if i % 20 ==0 :
                        print("test {}/{}".format(i,len(test_loader)))
                    im_seg_list, im_flow_list, gt = data
                    pred = net(im_seg_list.cuda(), im_flow_list.cuda())
                    out = torch.argmax(pred, dim=1)
                    out = out.squeeze().cpu().numpy()
                    gt = gt.squeeze().cpu().numpy()
                    miou_cal.update(gt, out)
                miou, miou_class = miou_cal.get_scores(return_class=True)
                miou_cal.reset()

            tblogger.add_scalar('miou_city/miou', miou, step + 1)
            for i in range(int(NUM_CLASS)):
                tblogger.add_scalar('miou_city/{}_miou'.format(label_name[i]), miou_class[i], step + 1)
            if miou > best_city_miou:
                best_city_miou = miou
                save_path = os.path.join(args.work_dirs, args.exp_name, 'src_only_one_stage_{}.pth'.format(args.model_name))
                torch.save(net.state_dict(), save_path)
            print('step:{} current city miou:{:.4f} best city miou:{:.4f}'.format(step + 1, miou, best_city_miou))
            logger.info('step:{} current city miou:{:.4f} best city miou:{:.4f}'.format(step + 1, miou, best_city_miou))


def poly_lr_scheduler(args, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
    init_lr = args.lr
    # iter = self.iter if iter is None else iter
    max_iter = args.train_iterations
    power = 0.9
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = 10 * new_lr
    return new_lr

if __name__ == '__main__':
    train()
