import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.deeplab_multi import DeeplabMulti as deeplabv2
from lib.model.flownet import FlowNets
from lib.model.warpnet import warp

class Accel101(nn.Module):
    def __init__(self, num_classes=15, weight_update=None, weight_reference=None, weight_flownet=None, multi_level=False):
        super(Accel101, self).__init__()
        self.multi_level = multi_level
        self.net_ref = deeplabv2(num_classes=num_classes, multi_level=self.multi_level)
        self.net_update = deeplabv2(num_classes=num_classes, multi_level=self.multi_level)
        self.merge = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0)

        self.flownet = FlowNets()
        self.warp = warp()

        self.weight_init(weight_update, weight_reference, weight_flownet)


        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=-1)

    def weight_init(self, weight_update, weight_reference, weight_flownet):
        weight = torch.load(weight_update, map_location='cpu')

        new_params = self.net_update.state_dict().copy()
        for i in weight:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = weight[i]
        self.net_update.load_state_dict(new_params, True)
        self.net_update.fix_backbone()

        weight = torch.load(weight_reference, map_location='cpu')
        new_params = self.net_ref.state_dict().copy()
        for i in weight:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = weight[i]
        self.net_ref.load_state_dict(new_params, True)
        self.net_ref.fix_backbone()

        weight = torch.load(weight_flownet, map_location='cpu')
        self.flownet.load_state_dict(weight, True)

        nn.init.xavier_normal_(self.merge.weight)
        self.merge.bias.data.fill_(0)
        print('pretrained weight loaded')

    def forward(self, im_seg_list, im_flow_list, gt=None, return_down_scale=False):
        n, c, t, h, w = im_seg_list.shape
        
        if self.multi_level:
            pred = self.net_ref(im_seg_list[:, :, 0, :, :])[0] # not use aux
        else:
            pred = self.net_ref(im_seg_list[:, :, 0, :, :])
        pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
        for i in range(t - 1):
            flow = self.flownet(torch.cat([im_flow_list[:, :, i + 1, :, :], im_flow_list[:, :, i, :, :]], dim=1))
            pred = self.warp(pred, flow)
        if self.multi_level:
            pred_update = self.net_update(im_seg_list[:, :, -1, :, :])[0] # not use aux
        else:
            pred_update = self.net_update(im_seg_list[:, :, -1, :, :])
        pred_update = F.interpolate(pred_update, scale_factor=2, mode='bilinear', align_corners=False)
        pred_merge = self.merge(torch.cat([pred, pred_update], dim=1))
        if return_down_scale:
            return pred_merge

        pred_merge = F.interpolate(pred_merge, scale_factor=4, mode='bilinear', align_corners=False)

        if gt is not None:
            loss = self.criterion_semantic(pred_merge, gt)
            loss = loss.unsqueeze(0)
            return loss
        else:
            return pred_merge

class Accel101_SFC(nn.Module):
    def __init__(self, num_classes=15, weight_update=None, weight_reference=None, weight_flownet=None, weight_flownet_global=None, weight_SFM=None, multi_level=False, scale=1):
        super(Accel101_SFC, self).__init__()
        self.scale = scale
        self.multi_level = multi_level
        self.net_ref = deeplabv2(num_classes=num_classes, multi_level=self.multi_level)
        self.net_update = deeplabv2(num_classes=num_classes, multi_level=self.multi_level)
        self.merge = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0)

        self.flownet = FlowNets()
        self.flownet_global = FlowNets()
        self.SFM = FlowNets()
        self.warp = warp()

        self.weight_init(weight_update, weight_reference, weight_flownet, weight_flownet_global, weight_SFM)

        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=-1)

    def weight_init(self, weight_update, weight_reference, weight_flownet, weight_flownet_global, weight_SFM):
        weight = torch.load(weight_update, map_location='cpu')

        new_params = self.net_update.state_dict().copy()
        for i in weight:
            i_parts = i.split('.')
            if not i_parts[0] == 'layer5':
                new_params['.'.join(i_parts[0:])] = weight[i]
        self.net_update.load_state_dict(new_params, True)
        self.net_update.fix_backbone()

        weight = torch.load(weight_reference, map_location='cpu')
        new_params = self.net_ref.state_dict().copy()
        for i in weight:
            i_parts = i.split('.')
            if not i_parts[0] == 'layer5':
                new_params['.'.join(i_parts[0:])] = weight[i]
        self.net_ref.load_state_dict(new_params, True)
        self.net_ref.fix_backbone()

        weight = torch.load(weight_flownet, map_location='cpu')
        self.flownet.load_state_dict(weight, True)

        weight = torch.load(weight_flownet_global, map_location='cpu')
        self.flownet_global.load_state_dict(weight, True)

        weight = torch.load(weight_SFM, map_location='cpu')
        self.SFM.load_state_dict(weight, True)

        nn.init.xavier_normal_(self.merge.weight)
        self.merge.bias.data.fill_(0)
        print('pretrained weight loaded')

    def forward(self, im_seg_list, im_flow_list, im_seg_list0=None, im_flow_list0=None, gt=None, flow_consist=False):
        n, c, t, h, w = im_seg_list.shape
        
        if self.multi_level:
            pred = self.net_ref(im_seg_list[:, :, 0, :, :])[0] # not use aux
        else:
            pred = self.net_ref(im_seg_list[:, :, 0, :, :])
        pred = F.interpolate(pred, scale_factor=2, mode='bilinear', align_corners=False)
        for i in range(t - 1):
            flow = self.flownet(torch.cat([im_flow_list[:, :, i + 1, :, :], im_flow_list[:, :, i, :, :]], dim=1))
            pred = self.warp(pred, flow)
        if self.multi_level:
            pred_update = self.net_update(im_seg_list[:, :, -1, :, :])[0] # not use aux
        else:
            pred_update = self.net_update(im_seg_list[:, :, -1, :, :])
        pred_update = F.interpolate(pred_update, scale_factor=2, mode='bilinear', align_corners=False)
        pred_merge = self.merge(torch.cat([pred, pred_update], dim=1))
        pred_merge = F.interpolate(pred_merge, scale_factor=4, mode='bilinear', align_corners=False)
        
        if flow_consist:
            # first step: get t-1 merge pred
            pred0 = self.net_ref(im_seg_list0[:, :, 0, :, :])
            pred0 = F.interpolate(pred0, scale_factor=2, mode='bilinear', align_corners=False)
            flow0 = self.flownet(torch.cat([im_flow_list0[:, :, 1, :, :], im_flow_list0[:, :, 0, :, :]], dim=1))
            pred0 = self.warp(pred0, flow0)
            pred_update0 = self.net_update(im_seg_list0[:, :, -1, :, :])
            pred_update0 = F.interpolate(pred_update0, scale_factor=2, mode='bilinear', align_corners=False)
            pred_merge0 = self.merge(torch.cat([pred0, pred_update0], dim=1))
            pred_merge0 = F.interpolate(pred_merge0, scale_factor=4, mode='bilinear', align_corners=False)


            # get class merge flow_pred
            img0, img1 = im_flow_list[:, :, 0, :, :], im_flow_list[:, :, 1, :, :]
            pred_prob0 = F.softmax(pred_merge0* self.scale)
            pred_prob1 = F.softmax(pred_merge* self.scale)
            flow_label = self.flownet_global(torch.cat([img1, img0], dim=1))
            flow_pred = torch.zeros_like(flow_label)
            for j in range(15):
                img0_class = pred_prob0[:, j, :, :].squeeze() * img0
                img1_class = pred_prob1[:, j, :, :].squeeze() * img1
                flow_class = self.SFM(torch.cat([img1_class, img0_class], dim=1))
                flow_pred += flow_class
            if gt is not None:
                loss = self.criterion_semantic(pred_merge, gt)
                loss = loss.unsqueeze(0)
                return loss, flow_label, flow_pred 
            else:
                return flow_label, flow_pred 
        else:

            if gt is not None:
                loss = self.criterion_semantic(pred_merge, gt)
                loss = loss.unsqueeze(0)
                return loss
            else:
                return pred_merge

