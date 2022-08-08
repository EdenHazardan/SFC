import os
import sys
import cv2
import random
import numpy as np
import shutil
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as ttransforms

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


class viper_video_dataset(Dataset):
    def __init__(self, split='train', distance=2):
        self.data_path = '/home/gaoy/SFC/data'
        self.im_path = os.path.join(self.data_path, 'VIPER', split, 'img')
        self.gt_path = os.path.join(self.data_path, 'VIPER', split, 'cls')
        self.split = split
        self.distance = distance
        
        # Viper to City Label map
        self.id_to_trainid = {3: 0, 4: 1, 9: 2, 11: 3, 13: 4, 14: 5, 7: 6, 8: 6, 6: 7, 2: 8, 20: 9, 24: 10, 27: 11, 26: 12, 23: 13, 22: 14}
        self.ignore_ego_vehicle = True

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'viper_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'viper_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)

            frame_id = int(line[-9:-4])
            frame_prefix = line[:7]
            tmp = []
            for i in range(self.distance):
                name = '{}_{:05d}'.format(frame_prefix, frame_id - self.distance + 1 + i)+'.jpg'
                tmp.append(name)
            self.im_name.append(tmp)

    def __len__(self):
        return len(self.gt_name)

    def transform(self, im_list):
        im_seg_list = []
        im_flow_list = []
        for i in range(len(im_list)):
            im = im_list[i]
            im_flow = self._flow_transform(im.copy())
            im_flow_list.append(im_flow)

            im_seg = self._img_transform(im.copy())
            im_seg_list.append(im_seg)

        return im_seg_list, im_flow_list

    def _img_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image

    def _flow_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return image

    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1280, 720), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        # ignore ego vehicle, following DA-VSN
        if self.ignore_ego_vehicle:
            lbl_car = label == 24
            ret, lbs, stats, centroid = cv2.connectedComponentsWithStats(np.uint8(lbl_car))
            lb_vg = lbs[-1, lbs.shape[1] // 2]
            if lb_vg > 0:
                label[lbs == lb_vg] = 0
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, idx):
        im_name_list = self.im_name[idx]
        im_list = []
        for i in range(len(im_name_list)):
            name = im_name_list[i]
            image_path = os.path.join(self.im_path, name)
            im = Image.open(image_path).convert("RGB")
            im_list.append(im)
        gt_path = os.path.join(self.gt_path, name.replace('jpg','png'))
        gt = Image.open(gt_path)

        # data normalization
        im_seg_list, im_flow_list = self.transform(im_list)
        gt = self._mask_transform(gt)

        for i in range(len(im_list)):
            im_seg_list[i] = im_seg_list[i].float().unsqueeze(1)
            im_flow_list[i] = im_flow_list[i].float().unsqueeze(1)
        im_seg_list = torch.cat(im_seg_list, dim=1)
        im_flow_list = torch.cat(im_flow_list, dim=1)

        return im_seg_list, im_flow_list, gt

class viper_video_dataset_SFM(Dataset):
    def __init__(self, split='train', distance=2):
        self.data_path = '/home/gaoy/SFC/data'
        self.im_path = os.path.join(self.data_path, 'VIPER', split, 'img')
        self.gt_path = os.path.join(self.data_path, 'VIPER', split, 'cls')
        self.split = split
        self.distance = distance
        
        # Viper to City Label map
        self.id_to_trainid = {3: 0, 4: 1, 9: 2, 11: 3, 13: 4, 14: 5, 7: 6, 8: 6, 6: 7, 2: 8, 20: 9, 24: 10, 27: 11, 26: 12, 23: 13, 22: 14}
        self.ignore_ego_vehicle = True

        # data process
        self.get_list()


    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'viper_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'viper_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)

            frame_id = int(line[-9:-4])
            frame_prefix = line[:7]
            tmp = []
            for i in range(self.distance):
                name = '{}_{:05d}'.format(frame_prefix, frame_id - self.distance + 1 + i)+'.jpg'
                tmp.append(name)
            self.im_name.append(tmp)

    def __len__(self):
        return len(self.gt_name)

    def transform(self, im_list):
        im_flow_list = []
        for i in range(len(im_list)):
            im = im_list[i]
            im_flow = self._flow_transform(im.copy())
            im_flow_list.append(im_flow)

        return im_flow_list

    def _flow_transform(self, image):
        image = image.resize((1280, 720), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return image

    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target

    def id2trainId(self, label, ignore_label=-1):
        # ignore ego vehicle
        if self.ignore_ego_vehicle:
            lbl_car = label == 24
            ret, lbs, stats, centroid = cv2.connectedComponentsWithStats(np.uint8(lbl_car))
            lb_vg = lbs[-1, lbs.shape[1] // 2]
            if lb_vg > 0:
                label[lbs == lb_vg] = 0
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy

    def __getitem__(self, idx):
        im_name_list = self.im_name[idx]
        im_list = []
        for i in range(len(im_name_list)):
            name = im_name_list[i]
            image_path = os.path.join(self.im_path, name)
            im = Image.open(image_path).convert("RGB")
            im_list.append(im)
        gt_path = os.path.join(self.gt_path, name.replace('jpg','png'))
        frame_id = int(gt_path[-9:-4])
        frame_prefix = gt_path[:-10]
        gt0_path = frame_prefix + '_' + '{:05d}.png'.format(frame_id - 1)

        gt = Image.open(gt_path)
        gt0 = Image.open(gt0_path)
        gt = gt.resize((1280, 720), Image.NEAREST)
        gt0 = gt0.resize((1280, 720), Image.NEAREST)
        gt_down4x = gt.resize((320, 180), Image.NEAREST)

        # data normalization
        im_flow_list = self.transform(im_list)
        gt0 = self._mask_transform(gt0)
        gt = self._mask_transform(gt)
        gt_down4x = self._mask_transform(gt_down4x)

        for i in range(len(im_list)):
            im_flow_list[i] = im_flow_list[i].float().unsqueeze(1)
        im_flow_list = torch.cat(im_flow_list, dim=1)

        return im_flow_list, gt0, gt, gt_down4x

