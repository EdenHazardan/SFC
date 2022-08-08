import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.flownet import FlowNets
from lib.model.warpnet import warp

NUM_CLASS = 15

class SFM_train(nn.Module):
    def __init__(self, weight_flownet_global=None, weight_SFM=None):
        super(SFM_train, self).__init__()
        self.flownet_global = FlowNets()
        self.SFM = FlowNets()

        self.weight_init(weight_flownet_global, weight_SFM)

        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=-1)

    def weight_init(self, weight_flownet_global, weight_SFM):
        weight = torch.load(weight_flownet_global, map_location='cpu')
        self.flownet_global.load_state_dict(weight, True)

        weight = torch.load(weight_SFM, map_location='cpu')
        self.SFM.load_state_dict(weight, True)

        print('pretrained weight loaded')

    def forward(self, im_flow_list, gt0, gt1):
        n, c, t, h, w = im_flow_list.shape
        
        flow_label = self.flownet_global(torch.cat([im_flow_list[:, :, 1, :, :], im_flow_list[:, :, 0, :, :]], dim=1))

        img0, img1 = im_flow_list[:, :, 0, :, :], im_flow_list[:, :, 1, :, :]
        class_list = torch.cat([torch.unique(gt0), torch.unique(gt1)], dim=0)
        zeros = torch.zeros_like(gt1)
        ones = torch.ones_like(gt1)
        flow_pred = torch.zeros_like(flow_label)
        for i in range(NUM_CLASS):
            if i in class_list:
                gt_map0 = torch.where(gt0==i, ones, zeros)  # class mask
                gt_map1 = torch.where(gt1==i, ones, zeros)
                input0 = img0 * gt_map0
                input1 = img1 * gt_map1
                class_pred = self.SFM(torch.cat([input1, input0], dim=1))
                flow_pred += class_pred
            
        return flow_label, flow_pred

    def set_train(self):
        self.flownet_global.eval()
        self.SFM.train()
