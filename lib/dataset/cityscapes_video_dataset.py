
import os
import sys
import cv2
import random
import shutil
import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TTF
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as ttransforms

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# Labels
ignore_label = -1
cityscapes_id_to_trainid = {
    -1: ignore_label,
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: ignore_label,
    5: ignore_label,
    6: ignore_label,
    7: 0,
    8: 1,
    9: ignore_label,
    10: ignore_label,
    11: 2,
    12: ignore_label,
    13: 3,
    14: ignore_label,
    15: ignore_label,
    16: ignore_label,
    17: ignore_label,
    18: ignore_label,
    19: 4,
    20: 5,
    21: 6,
    22: 7,
    23: 8,
    24: 9,
    25: 9,
    26: 10,
    27: 11,
    28: 12,
    29: ignore_label,
    30: ignore_label,
    31: ignore_label,
    32: 13,
    33: 14
}

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

class cityscapes_video_dataset(Dataset):
    def __init__(self, split='train', distance=2):
        self.data_path = '/home/gaoy/SFC/data'
        self.im_path = os.path.join(self.data_path, 'Cityscapes', 'leftImg8bit_sequence', split)
        self.gt_path = os.path.join(self.data_path, 'Cityscapes', 'gtFine', split)
        self.split = split
        self.distance = distance
        self.crop_size = (256, 512)
        
        # Viper to City Label map
        self.id_to_trainid = cityscapes_id_to_trainid

        self.get_list()

    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'city_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'city_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)

            frame_id = int(line[-6:])
            filename = line.split("_")[1]
            
            if self.split == 'train':
                frame_prefix = line[6:-7]
            else:
                frame_prefix = line[4:-7]
            tmp = []
            for i in range(self.distance):
                name = '{}/{}_{:06d}'.format(filename, frame_prefix, frame_id - self.distance + 1 + i)
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
        image = image.resize((1024, 512), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image

    def _flow_transform(self, image):
        image = image.resize((1024, 512), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return image

    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1024, 512), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy


    def __getitem__(self, idx):
        im_name_list = self.im_name[idx]
        im_list = []
        for i in range(len(im_name_list)):
            name = im_name_list[i]
            image_path = os.path.join(os.path.join(self.im_path, '{}_leftImg8bit.png'.format(name)))
            im = Image.open(image_path).convert("RGB")
            im_list.append(im)
    
        gt_path = os.path.join(os.path.join(self.gt_path, '{}_gtFine_labelIds.png'.format(name)))
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

class cityscapes_video_dataset_SFC(Dataset):
    def __init__(self, split='train', distance=2):
        self.data_path = '/home/gaoy/SFC/data'
        self.im_path = os.path.join(self.data_path, 'Cityscapes', 'leftImg8bit_sequence', split)
        self.gt_path = os.path.join(self.data_path, 'Cityscapes', 'gtFine', split)
        self.split = split
        self.distance = distance

        # Viper to City Label map
        self.id_to_trainid = cityscapes_id_to_trainid

        self.get_list()

    def get_list(self):
        self.im_name = []
        self.gt_name = []

        if self.split == 'train':
            list_path = os.path.join(self.data_path, 'city_list', 'train.txt')
            print("load list path : ",list_path)
        else:
            list_path = os.path.join(self.data_path, 'city_list', 'val.txt')
        with open(list_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            self.gt_name.append(line)

            frame_id = int(line[-6:])
            filename = line.split("_")[1]
            
            if self.split == 'train':
                frame_prefix = line[6:-7]
            else:
                frame_prefix = line[4:-7]
            tmp = []
            for i in range(3):
                name = '{}/{}_{:06d}'.format(filename, frame_prefix, frame_id - 3 + 1 + i)
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
        image = image.resize((1024, 512), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)
        return new_image

    def _flow_transform(self, image):
        image = image.resize((1024, 512), Image.BICUBIC)
        image = np.asarray(image, np.float32)
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        return image

    def _mask_transform(self, gt_image):
        gt_image = gt_image.resize((1024, 512), Image.NEAREST)
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target.astype(np.int64)).long()

        return target


    def id2trainId(self, label, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy


    def __getitem__(self, idx):
        im_name_list = self.im_name[idx]
        im_list = []
        for i in range(len(im_name_list)):
            name = im_name_list[i]
            image_path = os.path.join(os.path.join(self.im_path, '{}_leftImg8bit.png'.format(name)))
            im = Image.open(image_path).convert("RGB")
            im_list.append(im)
    
        gt_path = os.path.join(os.path.join(self.gt_path, '{}_gtFine_labelIds.png'.format(name)))
        gt = Image.open(gt_path)

        # data normalization
        im_seg_list, im_flow_list = self.transform(im_list)
        gt = self._mask_transform(gt)

        for i in range(len(im_list)):
            im_seg_list[i] = im_seg_list[i].float().unsqueeze(1)
            im_flow_list[i] = im_flow_list[i].float().unsqueeze(1)
        im_seg_list1 = [im_seg_list[1], im_seg_list[2]]
        im_seg_list0 = [im_seg_list[0], im_seg_list[1]]
        im_flow_list1 = [im_flow_list[1], im_flow_list[2]]
        im_flow_list0 = [im_flow_list[0], im_flow_list[1]]
        im_seg_list1 = torch.cat(im_seg_list1, dim=1)
        im_flow_list1 = torch.cat(im_flow_list1, dim=1)
        im_seg_list0 = torch.cat(im_seg_list0, dim=1)
        im_flow_list0 = torch.cat(im_flow_list0, dim=1)

        return im_seg_list1, im_flow_list1, im_seg_list0, im_flow_list0, gt
