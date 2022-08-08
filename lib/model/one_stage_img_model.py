import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.deeplab_multi import DeeplabMulti as deeplabv2
from lib.model.flownet import FlowNets

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

class src_only_one_stage_model(nn.Module):
    def __init__(self, num_classes=15, weight_deeplabv2_res101=None, multi_level=True):
        super(src_only_one_stage_model, self).__init__()
        self.multi_level = multi_level
        self.segnet = deeplabv2(num_classes=num_classes, multi_level=self.multi_level)

        self.weight_init(weight_deeplabv2_res101)

        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=-1)

    def weight_init(self, weight_deeplabv2_res101):
        weight = torch.load(weight_deeplabv2_res101, map_location='cpu')

        new_params = self.segnet.state_dict().copy()
        for i in weight:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = weight[i]
        self.segnet.load_state_dict(new_params, True)

        print('pretrained weight loaded')

    def forward(self, im_seg_list, im_flow_list, gt=None, SFC=False):
        n, c, t, h, w = im_seg_list.shape
        if self.multi_level:
            # obtain segmentation predictions
            pred_c_1, pred_c_2 = self.segnet(im_seg_list[:, :, 1, :, :])

            pred_c_1 = F.interpolate(pred_c_1, scale_factor=8, mode='bilinear', align_corners=False)
            pred_c_2 = F.interpolate(pred_c_2, scale_factor=8, mode='bilinear', align_corners=False)

            if gt is not None:
                loss_1 = self.criterion_semantic(pred_c_1, gt)
                loss_2 = self.criterion_semantic(pred_c_2, gt)
                loss_1 = loss_1.unsqueeze(0)
                loss_2 = loss_2.unsqueeze(0)
                loss = loss_1 + loss_2 * 0.1 
                return loss
            else:
                return pred_c_1

        else:
            # obtain segmentation predictions
            pred_c = self.segnet(im_seg_list[:, :, 1, :, :])
            pred_c = F.interpolate(pred_c, scale_factor=8, mode='bilinear', align_corners=False)
        
            if gt is not None:
                loss = self.criterion_semantic(pred_c, gt)
                loss = loss.unsqueeze(0)
                return loss
            else:
                return pred_c

class SFC_one_stage_model(nn.Module):
    def __init__(self, num_classes=15, weight_deeplabv2_res101=None, weight_flownet_global=None, weight_SFM=None, multi_level=True, scale=100):
        super(SFC_one_stage_model, self).__init__()
        self.scale = scale
        self.multi_level = multi_level
        self.segnet = deeplabv2(num_classes=num_classes, multi_level=self.multi_level)

        self.flownet_global = FlowNets()
        self.SFM = FlowNets()

        self.weight_init(weight_deeplabv2_res101, weight_flownet_global, weight_SFM)

        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=-1)

    def weight_init(self, weight_deeplabv2_res101, weight_flownet_global, weight_SFM):
        weight = torch.load(weight_deeplabv2_res101, map_location='cpu')

        new_params = self.segnet.state_dict().copy()
        for i in weight:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = weight[i]
        self.segnet.load_state_dict(new_params, True)

        weight = torch.load(weight_flownet_global, map_location='cpu')
        self.flownet_global.load_state_dict(weight, True)

        weight = torch.load(weight_SFM, map_location='cpu')
        self.SFM.load_state_dict(weight, True)

        print('pretrained weight loaded')



    def forward(self, im_seg_list, im_flow_list, gt=None, SFC=False):
        n, c, t, h, w = im_seg_list.shape
        if self.multi_level:
            # obtain segmentation predictions
            pred_k_1, pred_k_2 = self.segnet(im_seg_list[:, :, 0, :, :])
            pred_c_1, pred_c_2 = self.segnet(im_seg_list[:, :, 1, :, :])

            pred_k_1 = F.interpolate(pred_k_1, scale_factor=8, mode='bilinear', align_corners=False)
            pred_c_1 = F.interpolate(pred_c_1, scale_factor=8, mode='bilinear', align_corners=False)
            pred_k_2 = F.interpolate(pred_k_2, scale_factor=8, mode='bilinear', align_corners=False)
            pred_c_2 = F.interpolate(pred_c_2, scale_factor=8, mode='bilinear', align_corners=False)
            
            if SFC:
                # get global optical flow as gt
                img0, img1 = im_flow_list[:, :, 0, :, :], im_flow_list[:, :, 1, :, :]
                flow_label = self.flownet_global(torch.cat([img1, img0], dim=1))
                flow_label = flow_label.detach()
                # adjust softmax operation by multiplying a scale factor
                pred_prob0 = F.softmax(pred_k_1* self.scale)
                pred_prob1 = F.softmax(pred_c_1* self.scale)
                flow_pred_1 = torch.zeros_like(flow_label)
                # extract class-activated regions 
                for j in range(15):
                    img0_class = pred_prob0[:, j, :, :].squeeze() * img0
                    img1_class = pred_prob1[:, j, :, :].squeeze() * img1
                    flow_class = self.SFM(torch.cat([img1_class, img0_class], dim=1))
                    flow_pred_1 += flow_class
                # class-global flow consistency
                loss_flow_1 = EPE(flow_pred_1, flow_label)

                # multi-scale SFC (the multi-scale is used widely in DVSS, e.g., DA-VSN, PixMatch)
                pred_prob0 = F.softmax(pred_k_2* self.scale)
                pred_prob1 = F.softmax(pred_c_2* self.scale)
                flow_pred_2 = torch.zeros_like(flow_label)
                for j in range(15):
                    img0_class = pred_prob0[:, j, :, :].squeeze() * img0
                    img1_class = pred_prob1[:, j, :, :].squeeze() * img1
                    flow_class = self.SFM(torch.cat([img1_class, img0_class], dim=1))
                    flow_pred_2 += flow_class
                loss_flow_2 = EPE(flow_pred_2, flow_label)
                loss_flow = loss_flow_1 + loss_flow_2 * 0.1
                return loss_flow 

            else:
                if gt is not None:
                    loss_1 = self.criterion_semantic(pred_c_1, gt)
                    loss_2 = self.criterion_semantic(pred_c_2, gt)
                    loss_1 = loss_1.unsqueeze(0)
                    loss_2 = loss_2.unsqueeze(0)
                    loss = loss_1 + loss_2 * 0.1 
                    return loss
                else:
                    return pred_c_1


        else:
            # obtain segmentation predictions
            pred_k = self.segnet(im_seg_list[:, :, 0, :, :])
            pred_c = self.segnet(im_seg_list[:, :, 1, :, :])

            pred_k = F.interpolate(pred_k, scale_factor=8, mode='bilinear', align_corners=False)
            pred_c = F.interpolate(pred_c, scale_factor=8, mode='bilinear', align_corners=False)
            
            if SFC:
                img0, img1 = im_flow_list[:, :, 0, :, :], im_flow_list[:, :, 1, :, :]
                # get global optical flow as gt
                flow_label = self.flownet_global(torch.cat([img1, img0], dim=1))
                flow_label = flow_label.detach()
                # adjust softmax operation by multiplying a scale factor
                pred_prob0 = F.softmax(pred_k* self.scale)
                pred_prob1 = F.softmax(pred_c* self.scale)
                flow_pred = torch.zeros_like(flow_label)
                # extract class-activated regions 
                for j in range(15):
                    img0_class = pred_prob0[:, j, :, :].squeeze() * img0
                    img1_class = pred_prob1[:, j, :, :].squeeze() * img1
                    flow_class = self.SFM(torch.cat([img1_class, img0_class], dim=1))
                    flow_pred += flow_class
                # class-global flow consistency
                loss_flow = EPE(flow_pred, flow_label)
                return loss_flow 

            else:
                if gt is not None:
                    loss = self.criterion_semantic(pred_c, gt)
                    loss = loss.unsqueeze(0)
                    return loss
                else:
                    return pred_c


