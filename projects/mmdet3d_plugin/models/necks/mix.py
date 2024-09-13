import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from torch.nn.parameter import Parameter

class channel_spatial_stage (nn.Module):
    def __init__(self, features):
        """ Constructor
        Args:
            features: 两个 feat concat 之后的通道数
        """
        super (channel_spatial_stage, self).__init__ ()
        reduction=16
        # self.M = M

        self.channels = features//2  # bev 和 voxel 分别的 feat  self.channels=C

        self.fc = nn.Sequential(
            nn.Linear(features, features//reduction),
            nn.ReLU(inplace=False),
            nn.Linear(features//reduction,self.channels),
            nn.Sigmoid()
        )

        self.spacial_leanring = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, 2C, 1, 1)
        x_bev,x_voxel = torch.split (x, self.channels, dim=1)
        fea_U = x  # (B, 2C, 1, 1)
        fea_S = fea_U.mean (-1).mean (-1)  # (B, 2C)

        attention_1 = self.fc (fea_S)   # (B, C)
        attention_1 = attention_1.unsqueeze (-1).unsqueeze (-1)  # (B, C, 1, 1)

        x_bev_1 = attention_1*x_bev

        x_voxel_1 = (1-attention_1)*x_voxel

        fea_U_1 = x_bev_1 + x_voxel_1
        fea_S_1 = self.spacial_leanring(fea_U_1)

        attention_2 = self.sigmoid(fea_S_1)
        x_bev_2 = attention_2*x_bev_1
        x_voxel_2 = (1-attention_2)*x_voxel_1

        x_fuse = x_bev_2 + x_voxel_2

        return x_fuse

@NECKS.register_module()
class SFA (BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super (SFA, self).__init__ ()

        self.mysk_7=channel_spatial_stage(features=in_channels)

        self.mix_channels = in_channels
        self.out_channels = out_channels

        self.mix_residual = nn.Sequential (
            nn.Conv2d (self.mix_channels//2, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d (self.out_channels),
            nn.ReLU (inplace=True),
            nn.Conv2d (self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d (self.out_channels))

        self.mix_shortcut = nn.Sequential (
            nn.Conv2d (self.mix_channels, self.out_channels, stride=stride, kernel_size=1, bias=False),
            nn.BatchNorm2d (self.out_channels))

        self.relu = nn.ReLU (inplace=True)

    def forward(self, inputs):
        result = self.mysk_7(inputs)
        result = self.relu (self.mix_residual (result) + self.mix_shortcut (inputs))
        return result

