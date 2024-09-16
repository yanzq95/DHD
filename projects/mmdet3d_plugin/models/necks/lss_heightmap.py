# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule,force_fp32
from mmdet3d.models.builder import NECKS
from ...ops import bev_pool_v2
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
from ..model_utils import DepthNet,HeightNet

@NECKS.register_module ()
class MGHS (BaseModule):
    def __init__(
            self,
            grid_config,
            input_size,
            downsample=16,
            in_channels=512,
            out_channels=64,
            heightnet_cfg=dict(),
            accelerate=False,
            sid=False,
            collapse_z=True,
            height_range=[-1.5, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
            height_interval = 0.5,
            mask_range = [-5,0,0.4,5], # h_min, thr1, thr2, h_max
            loss_height_weight = 1.0,
            mask_1_grid={
                'x': [-40, 40, 0.4],
                'y': [-40, 40, 0.4],
                'z': [-1, 2.2, 0.4],
                'depth': [1.0, 45.0, 0.5],  # (45-1)/0.5
            },
            mask_2_grid={
                'x': [-40, 40, 0.4],
                'y': [-40, 40, 0.4],
                'z': [2.2, 3.8, 0.4],
                'depth': [1.0, 45.0, 0.5],  # (45-1)/0.5
            },
            mask_3_grid={
                'x': [-40, 40, 0.4],
                'y': [-40, 40, 0.4],
                'z': [3.8, 5.4, 0.4],
                'depth': [1.0, 45.0, 0.5],  # (45-1)/0.5
            }

    ):
        super (MGHS, self).__init__ ()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos (**grid_config)
        self.sid = sid
        self.frustum = self.create_frustum (grid_config['depth'],
                                            input_size, downsample)  # (D, fH, fW, 3)  3:(u, v, d)  (88,16,44,3)
        self.accelerate = accelerate
        self.initial_flag = True

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.depth_net = nn.Conv2d (in_channels, self.D + self.out_channels, kernel_size=1, padding=0)

        self.H = len(height_range)

        self.height_net = HeightNet (
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            depth_channels=self.H,
            **heightnet_cfg)

        self.collapse_z = collapse_z

        self.height_range = height_range
        self.mask_range = mask_range

        self.height_interval = height_interval

        self.loss_height_weight = loss_height_weight

        self.mask_1_grid = mask_1_grid
        self.mask_2_grid = mask_2_grid
        self.mask_3_grid = mask_3_grid

    # Origin LSS
    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor ([cfg[0] for cfg in [x, y, z]])  # (min_x, min_y, min_z)
        self.grid_interval = torch.Tensor ([cfg[2] for cfg in [x, y, z]])  # (dx, dy, dz)
        self.grid_size = torch.Tensor ([(cfg[1] - cfg[0]) / cfg[2]
                                        for cfg in [x, y, z]])  # (Dx, Dy, Dz)

    # Origin LSS
    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        Returns:
            frustum: (D, fH, fW, 3)  3:(u, v, d)
        """
        H_in, W_in = input_size  # 256 x 704
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange (*depth_cfg, dtype=torch.float).view (-1, 1, 1).expand (-1, H_feat, W_feat)  # (D, fH, fW)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange (self.D).float ()
            depth_cfg_t = torch.tensor (depth_cfg).float ()
            d_sid = torch.exp (torch.log (depth_cfg_t[0]) + d_sid / (self.D - 1) *
                               torch.log ((depth_cfg_t[1] - 1) / depth_cfg_t[0]))
            d = d_sid.view (-1, 1, 1).expand (-1, H_feat, W_feat)

        x = torch.linspace (0, W_in - 1, W_feat, dtype=torch.float) \
            .view (1, 1, W_feat).expand (self.D, H_feat, W_feat)  # (D, fH, fW)
        y = torch.linspace (0, H_in - 1, H_feat, dtype=torch.float) \
            .view (1, H_feat, 1).expand (self.D, H_feat, W_feat)  # (D, fH, fW)

        return torch.stack ((x, y, d), -1)  # (D, fH, fW, 3)  3:(u, v, d)

    # Origin LSS
    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3

        points = self.frustum.to (sensor2ego) - post_trans.view (B, N, 1, 1, 1, 3)
        points = torch.inverse (post_rots).view (B, N, 1, 1, 1, 3, 3) \
            .matmul (points.unsqueeze (-1))

        # cam_to_ego
        points = torch.cat (
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:, :, :3, :3].matmul (torch.inverse (cam2imgs))
        points = combine.view (B, N, 1, 1, 1, 3, 3).matmul (points).squeeze (-1)
        points += sensor2ego[:, :, :3, 3].view (B, N, 1, 1, 1, 3)
        points = bda.view (B, 1, 1, 1, 1, 3,
                           3).matmul (points.unsqueeze (-1)).squeeze (-1)
        return points

    # Origin LSS
    def get_ego_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                     bda):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            sensor2ego (torch.Tensor): Transformation from camera coordinate system to  相机坐标系 -> 车辆坐标系
                ego coordinate system in shape (B, N_cams, 4, 4).
            ego2global (torch.Tensor): Translation from ego coordinate system to  车辆坐标系 -> 全局坐标系
                global coordinate system in shape (B, N_cams, 4, 4).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape  相机的内参
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in  数据增强后相机坐标系的旋转矩阵
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system  数据增强后相机的平移矩阵（相机在图像增强期间相对于初始姿态的平移）
                derived from image view augmentation in shape (B, N_cams, 3).
            bda (torch.Tensor): Transformation in bev. (B, 3, 3)

        Returns:
            torch.tensor: Point coordinates in shape (B, N, D, fH, fW, 3)  在激光雷达坐标系中视锥体顶点位置的坐标
        """
        B, N, _, _ = sensor2ego.shape
        # 1、抵消数据增强及预处理对像素的变化（S1、S2）
        # S1：相机坐标系 - 平移量（post_trans）
        # (D, fH, fW, 3) - (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)

        points = self.frustum.to (sensor2ego) - post_trans.view (B, N, 1, 1, 1, 3)
        # S2：应用旋转矩阵的逆矩阵
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        points = torch.inverse (post_rots).view (B, N, 1, 1, 1, 3, 3) \
            .matmul (points.unsqueeze (-1))

        # 2、相机坐标系 -> 车辆坐标系（S1、S2）
        # S1：先将像素坐标(u,v,d)变成齐次坐标(d*u,d*v,d)
        # (B, N, D, fH, fW, 3, 1)  3: (d*u, d*v, d)
        points = torch.cat ((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        # S2：相机坐标系到车辆坐标系的转换
        # sensor2ego[:, :, :3, :3]：相机到车辆坐标系的旋转矩阵
        # torch.inverse (cam2imgs)：相机内参矩阵的逆矩阵
        # combine：坐标系的变换矩阵
        combine = sensor2ego[:, :, :3, :3].matmul (torch.inverse (cam2imgs))
        # (B, N, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1)  --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = combine.view (B, N, 1, 1, 1, 3, 3).matmul (points).squeeze (-1)
        # (B, N, D, fH, fW, 3) + (B, N, 1, 1, 1, 3) --> (B, N, D, fH, fW, 3)
        points += sensor2ego[:, :, :3, 3].view (B, N, 1, 1, 1, 3)

        # 3、bev argument
        # (B, 1, 1, 1, 3, 3) @ (B, N, D, fH, fW, 3, 1) --> (B, N, D, fH, fW, 3, 1)
        # --> (B, N, D, fH, fW, 3)
        points = bda.view (B, 1, 1, 1, 1, 3, 3).matmul (points.unsqueeze (-1)).squeeze (-1)
        return points

    # Origin LSS
    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
        interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2 (coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )

        self.ranks_bev = ranks_bev.int ().contiguous ()
        self.ranks_feat = ranks_feat.int ().contiguous ()
        self.ranks_depth = ranks_depth.int ().contiguous ()
        self.interval_starts = interval_starts.int ().contiguous ()
        self.interval_lengths = interval_lengths.int ().contiguous ()

    # Origin LSS
    def voxel_pooling_v2(self, coor, depth, feat):
        """
        Args:
            coor: (B, N, D, fH, fW, 3)
            depth: (B, N, D, fH, fW)
            feat: (B, N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
        """
        ranks_bev, ranks_depth, ranks_feat, \
        interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2 (coor)
        # ranks_bev: (N_points, ),
        # ranks_depth: (N_points, ),
        # ranks_feat: (N_points, ),
        # interval_starts: (N_pillar, )
        # interval_lengths: (N_pillar, )
        if ranks_feat is None:
            print ('warning ---> no points within the predefined '
                   'bev receptive field')
            dummy = torch.zeros (size=[
                feat.shape[0], feat.shape[2],
                int (self.grid_size[2]),
                int (self.grid_size[1]),
                int (self.grid_size[0])
            ]).to (feat)  # (B, C, Dz, Dy, Dx)
            dummy = torch.cat (dummy.unbind (dim=2), 1)  # (B, C*Dz, Dy, Dx)
            return dummy

        feat = feat.permute (0, 1, 3, 4, 2)  # (B, N, fH, fW, C)
        bev_feat_shape = (depth.shape[0], int (self.grid_size[2]),
                          int (self.grid_size[1]), int (self.grid_size[0]),
                          feat.shape[-1])  # (B, Dz, Dy, Dx, C)
        bev_feat = bev_pool_v2 (depth, feat, ranks_depth, ranks_feat, ranks_bev,
                                bev_feat_shape, interval_starts,
                                interval_lengths)  # (B, C, Dz, Dy, Dx)
        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat (bev_feat.unbind (dim=2), 1)  # (B, C*Dz, Dy, Dx)
        return bev_feat

    # Origin LSS
    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.
        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).
        Returns:
            tuple[torch.tensor]:
                ranks_bev: Rank of the voxel that a point is belong to in shape (N_points, ),
                    rank介于(0, B*Dx*Dy*Dz-1).
                ranks_depth: Reserved index of points in the depth space in shape (N_Points),
                    rank介于(0, B*N*D*fH*fW-1).
                ranks_feat: Reserved index of points in the feature space in shape (N_Points),
                    rank介于(0, B*N*fH*fW-1).
                interval_starts: (N_pillar, )
                interval_lengths: (N_pillar, )
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range (
            0, num_points - 1, dtype=torch.int, device=coor.device)  # (B*N*D*H*W, ), [0, 1, ..., B*N*D*fH*fW-1]
        ranks_feat = torch.range (
            0, num_points // D - 1, dtype=torch.int, device=coor.device)  # [0, 1, ...,B*N*fH*fW-1]
        ranks_feat = ranks_feat.reshape (B, N, 1, H, W)
        ranks_feat = ranks_feat.expand (B, N, D, H, W).flatten ()  # (B*N*D*fH*fW, )

        # convert coordinate into the voxel space
        # ((B, N, D, fH, fW, 3) - (3, )) / (3, ) --> (B, N, D, fH, fW, 3)   3:(x, y, z)  grid coords.
        coor = ((coor - self.grid_lower_bound.to (coor)) /
                self.grid_interval.to (coor))
        coor = coor.long ().view (num_points, 3)  # (B, N, D, fH, fW, 3) --> (B*N*D*fH*fW, 3)
        # (B, N*D*fH*fW) --> (B*N*D*fH*fW, 1)
        batch_idx = torch.range (0, B - 1).reshape (B, 1). \
            expand (B, num_points // B).reshape (num_points, 1).to (coor)
        coor = torch.cat ((coor, batch_idx), 1)  # (B*N*D*fH*fW, 4)   4: (x, y, z, batch_id)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
        if len (kept) == 0:
            return None, None, None, None, None

        # (N_points, 4), (N_points, ), (N_points, )
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]

        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
                self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
        ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort ()
        # (N_points, ), (N_points, ), (N_points, )
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones (
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where (kept)[0].int ()
        if len (interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like (interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int ().contiguous (), ranks_depth.int ().contiguous (
        ), ranks_feat.int ().contiguous (), interval_starts.int ().contiguous (
        ), interval_lengths.int ().contiguous ()

    # Origin LSS
    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_ego_coor (*input[1:7])  # (B, N, D, fH, fW, 3)
            self.init_acceleration_v2 (coor)
            self.initial_flag = False

    def view_transform_core(self, input, depth, tran_feat):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4) 传感器坐标系 -> 车辆坐标系 变换矩阵
                ego2globals: (B, N, 4, 4) 车辆坐标系 -> 全局坐标系 变换矩阵
                intrins:     (B, N, 3, 3) 相机内参
                post_rots:   (B, N, 3, 3) 后处理旋转参数
                post_trans:  (B, N, 3)    后处理平移参数
                bda_rot:  (B, 3, 3)       bev_data_argumentation
            depth:  (B*N, D, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C*Dz(=1), Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        B, N, C, H, W = input[0].shape  # fH=16,fW=44
        # Lift-Splat
        coor = self.get_ego_coor (*input[1:7])  # (B, N, D, fH, fW, 3)  # 获取车辆坐标系坐标
        # depth part: voxel pooling
        bev_feat = self.voxel_pooling_v2 (
            coor, depth.view (B, N, self.D, H, W),
            tran_feat.view (B, N, self.out_channels, H, W))  # (B, C*Dz(=1), Dy, Dx)

        return bev_feat, depth

    def view_transform(self, input, depth, tran_feat, height):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, C, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            height: (B*N, H, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        self.grid_config = {
            'x': [-40, 40, 0.4],
            'y': [-40, 40, 0.4],
            'z': [-1, 5.4, 6.4],
            'depth': [1.0, 45.0, 0.5],  # (45-1)/0.5
        }
        self.create_grid_infos (**self.grid_config)  
        bev_feat, depth_feat = self.view_transform_core (input, depth, tran_feat)  

        height_map = self.height_feature_to_height_map(height,self.height_range)  # (BxN, fH, fW)
        mask_1,mask_2,mask_3 = self.create_mask_3(height_map,h_min=self.mask_range[0],thr1=self.mask_range[1],thr2=self.mask_range[2],h_max=self.mask_range[3])  # (BxN, fH, fW)
        mask_1_expanded = mask_1.unsqueeze(1).expand_as(tran_feat)  # (BxN, C, fH, fW)
        mask_2_expanded = mask_2.unsqueeze(1).expand_as(tran_feat)
        mask_3_expanded = mask_3.unsqueeze(1).expand_as(tran_feat)

        masked_feat_1 = tran_feat * mask_1_expanded # (BxN, C, fH, fW)
        masked_feat_2 = tran_feat * mask_2_expanded
        masked_feat_3 = tran_feat * mask_3_expanded
        
        # L
        self.grid_config = self.mask_1_grid
        self.create_grid_infos (**self.grid_config)  
        bev_feat_masked_1, _ = self.view_transform_core (input, depth, masked_feat_1) 

        # M
        self.grid_config = self.mask_2_grid
        self.create_grid_infos (**self.grid_config) 
        bev_feat_masked_2, _ = self.view_transform_core (input, depth, masked_feat_2)  

        # H
        self.grid_config = self.mask_3_grid
        self.create_grid_infos (**self.grid_config)  
        bev_feat_masked_3, _ = self.view_transform_core (input, depth, masked_feat_3)  
    
        return bev_feat, depth_feat, height, bev_feat_masked_1,bev_feat_masked_2,bev_feat_masked_3

    def forward(self, input, stereo_metas=None):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]
        B, N, C, H, W = x.shape
        x = x.view (B * N, C, H, W)  # (B*N, C_in, fH, fW)
        # (B*N, C_in, fH, fW) --> (B*N, D+C, fH, fW)
        x_d = self.depth_net (x)
        depth_digit = x_d[:, :self.D, ...]  # (B*N, D, fH, fW)
        tran_feat_d = x_d[:, self.D:self.D + self.out_channels, ...]  # (B*N, C, fH, fW) 
        depth = depth_digit.softmax (dim=1) # (B*N, D, fH, fW)

        x_h = self.height_net(x, mlp_input, stereo_metas)
        height_digit = x_h[:, :self.H, ...]
        height = height_digit.softmax (dim=1) # (B*N, H, fH, fW)
        return self.view_transform (input, depth, tran_feat_d,height)


    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        """
        Args:
            sensor2ego: (B, N_views=6, 4, 4)
            ego2global: (B, N_views=6, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
        Returns:
            mlp_input: (B, N_views, 27)
        """
        B, N, _, _ = sensor2ego.shape
        bda = bda.view (B, 1, 3, 3).repeat (1, N, 1, 1)  # (B, 3, 3) --> (B, N, 3, 3)
        mlp_input = torch.stack ([
            intrin[:, :, 0, 0],  # fx
            intrin[:, :, 1, 1],  # fy
            intrin[:, :, 0, 2],  # cx
            intrin[:, :, 1, 2],  # cy
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2]
        ], dim=-1)  # (B, N_views, 15)
        sensor2ego = sensor2ego[:, :, :3, :].reshape (B, N, -1)
        mlp_input = torch.cat ([mlp_input, sensor2ego], dim=-1)  # (B, N_views, 27)
        return mlp_input

    def height_feature_to_height_map(self,height_feature,height_range):
        """
        Args:
            height_feature: (BxN,H,fH,fW)
            height_range: list (...) 长度=H
        Returns:
            height_map: (BxN,fH,fW)
        """
        if len (height_feature.shape) != 4:
            raise ValueError ("Input tensor must have 4 dimensions (BxN, H, fH, fW)")
        # 获取每个向量中最大值的索引
        height_indices = torch.argmax(height_feature, dim=1)  # (BxN,fH, fW)
        # 将索引映射到高度范围
        height_range_tensor = torch.tensor(height_range, device=height_feature.device)
        heights = height_range_tensor[height_indices]  # (BxN, fH, fW)
        return heights

    def create_mask_3(self, input_tensor, h_min, thr1, thr2, h_max):
        """

        Args:
            input_tensor: (BxN,H,W)
            thr1:
            thr2:

        Returns: bool mask
            mask_1: [h_min, thr1) -- low
            mask_2: [thr1 , thr2) -- mid
            mask_3: [thr2, h_max) -- high
        """
        # h_min = -5
        # h_max = 5

        mask_1 = (input_tensor >= h_min) & (input_tensor < thr1)
        mask_2 = (input_tensor >= thr1) & (input_tensor < thr2) 
        mask_3 = (input_tensor >= thr2) & (input_tensor < h_max)
        return mask_1, mask_2, mask_3

    def downsample_sparse_map(self, height_maps, downsample_factor=16):
        """
        对给定的稀疏高度图集合进行下采样并提取每个下采样块的最小高度值，
        忽略零值（空值），获得稠密的结果。
        参数：
        - height_maps: 形状为 (B, N, H, W) 的稀疏高度图集合
        - downsample_factor: 下采样倍数
        返回：
        - 下采样后的稠密高度图集合，形状为 (B, N, H // downsample_factor, W // downsample_factor)
        """
        B, N, H, W = height_maps.shape
        assert H % downsample_factor == 0, "Height must be divisible by downsample factor"
        assert W % downsample_factor == 0, "Width must be divisible by downsample factor"
        height_maps_tmp = torch.where (height_maps == 0.0, 1e5 * torch.ones_like (height_maps), height_maps)
        height_maps_tmp = height_maps_tmp.view (
            B, N, H // downsample_factor, downsample_factor,
                  W // downsample_factor, downsample_factor
        )
        height_maps_tmp = height_maps_tmp.permute (0, 1, 2, 4, 3, 5).contiguous ()
        height_maps_tmp = height_maps_tmp.view (B, N, -1, downsample_factor * downsample_factor)
        height_maps_downsampled = torch.min (height_maps_tmp, dim=-1).values
        height_maps_downsampled = torch.where (height_maps_downsampled == 1e5,
                                               torch.zeros_like (height_maps_downsampled),
                                               height_maps_downsampled)
        height_maps_downsampled = height_maps_downsampled.view (B, N, H // downsample_factor, W // downsample_factor)

        return height_maps_downsampled

    # TODO:  LOSS
    @force_fp32 ()
    def get_height_loss(self,gt_depth,gt_height,height):
        """
        Args:
            gt_depth: (B, N, H, W)
            gt_height: (B, N, H, W) LiDAR点投影到camera坐标系下的gt, H=256,W=704
            height: (B*N, H, fH, fW) 经过heightnet下采样后的高度feat
        Returns:
            loss_height
        """
        B, N, fH, fW = gt_height.shape
        height_labels = self.get_downsampled_gt_height (gt_height)  # (B*N*fH*fW, H)
        depth_labels = self.get_downsampled_gt_depth (gt_depth)  # (B*N*fH*fW, D)

        fg_mask = torch.max (depth_labels, dim=1).values > 0.0 # bool mask

        height_preds = height.permute(0,2,3,1).contiguous ().view (-1, self.H)  # (B*N*fH*fW, H)

        height_labels = height_labels[fg_mask]
        height_preds = height_preds[fg_mask]

        with autocast (enabled=False):
            height_loss = F.binary_cross_entropy (
                height_preds,
                height_labels,
                reduction='none'
            ).sum () / max (1.0, fg_mask.sum ())
        return self.loss_height_weight * height_loss


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: (B, N_views, img_h, img_w)
        Output:
            gt_depths: (B*N_views*fH*fW, D)
        """
        B, N, H, W = gt_depths.shape
        # (B*N_views, fH, downsample, fW, downsample, 1)
        gt_depths = gt_depths.view (B * N,
                                    H // self.downsample, self.downsample,
                                    W // self.downsample, self.downsample,
                                    1)
        # (B*N_views, fH, fW, 1, downsample, downsample)
        gt_depths = gt_depths.permute (0, 1, 3, 5, 2, 4).contiguous ()
        # (B*N_views*fH*fW, downsample, downsample)
        gt_depths = gt_depths.view (-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where (gt_depths == 0.0,
                                     1e5 * torch.ones_like (gt_depths),
                                     gt_depths)
        gt_depths = torch.min (gt_depths_tmp, dim=-1).values
        # (B*N_views, fH, fW)
        gt_depths = gt_depths.view (B * N, H // self.downsample, W // self.downsample)

        if not self.sid:
            # (D - (min_dist - interval_dist)) / interval_dist
            # = (D - min_dist) / interval_dist + 1
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log (gt_depths) - torch.log (
                torch.tensor (self.grid_config['depth'][0]).float ())
            gt_depths = gt_depths * (self.D - 1) / torch.log (
                torch.tensor (self.grid_config['depth'][1] - 1.).float () /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.

        gt_depths = torch.where ((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                 gt_depths, torch.zeros_like (gt_depths))  # (B*N_views, fH, fW)
        gt_depths = F.one_hot (
            gt_depths.long (), num_classes=self.D + 1).view (-1, self.D + 1)[:, 1:]  # (B*N_views*fH*fW, D)
        return gt_depths.float ()


    def get_downsampled_gt_height(self, gt_height):
        """
        Input:
            gt_height: (B, N_views, img_h, img_w)
        Output:
            gt_height: (B*N_views*fH*fW, D)
        """
        B, N, H, W = gt_height.shape
        # (B*N_views, fH, downsample, fW, downsample, 1)
        gt_height = gt_height.view (B * N,
                                    H // self.downsample, self.downsample,
                                    W // self.downsample, self.downsample,
                                    1)
        # (B*N_views, fH, fW, 1, downsample, downsample)
        gt_height = gt_height.permute (0, 1, 3, 5, 2, 4).contiguous ()
        # (B*N_views*fH*fW, downsample, downsample)
        gt_height = gt_height.view (-1, self.downsample * self.downsample)
        gt_height_tmp = torch.where (gt_height == 0.0,1e5 * torch.ones_like (gt_height),gt_height)
        gt_height = torch.min (gt_height_tmp, dim=-1).values
        # (B*N_views, fH, fW)
        gt_height = gt_height.view (B * N, H // self.downsample, W // self.downsample)
        # import pdb;pdb.set_trace()
        # (D - (min_dist - interval_dist)) / interval_dist
        # = (D - min_dist) / interval_dist + 1
        gt_height = (gt_height - self.height_range[0]) / self.height_interval

        gt_height = torch.where ((gt_height < self.H + 1) & (gt_height >= 0.0),
                                 gt_height, torch.zeros_like (gt_height))  # (B*N_views, fH, fW)

        gt_height = F.one_hot (gt_height.long (), num_classes=self.H + 1).view (-1, self.H + 1)[:, 1:]  # (B*N_views*fH*fW, H)

        return gt_height.float ()


@NECKS.register_module()
class MGHS_Depth(MGHS):
    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(MGHS_Depth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(
            in_channels=self.in_channels,
            mid_channels=self.in_channels,
            context_channels=self.out_channels,
            depth_channels=self.D,
            **depthnet_cfg)

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        """
        Args:
            sensor2ego: (B, N_views=6, 4, 4)
            ego2global: (B, N_views=6, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
        Returns:
            mlp_input: (B, N_views, 27)
        """
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)   # (B, 3, 3) --> (B, N, 3, 3)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],     # fx
            intrin[:, :, 1, 1],     # fy
            intrin[:, :, 0, 2],     # cx
            intrin[:, :, 1, 2],     # cy
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2]
        ], dim=-1)      # (B, N_views, 15)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)      # (B, N_views, 27)
        return mlp_input

    def forward(self, input, stereo_metas=None):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N_views, 3, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
                mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)      # (B*N_views, C, fH, fW)
        x_d = self.depth_net(x, mlp_input, stereo_metas)      # (B*N_views, D+C_context, fH, fW)
        depth_digit = x_d[:, :self.D, ...]    # (B*N_views, D, fH, fW)
        tran_feat = x_d[:, self.D:self.D + self.out_channels, ...]    # (B*N_views, C_context, fH, fW)
        depth = depth_digit.softmax(dim=1)  # (B*N_views, D, fH, fW)

        x_h = self.height_net(x,mlp_input,stereo_metas=None)
        height_digit = x_h[:,:self.H,...]
        height = height_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat,height)

    def view_transform(self, input, depth, tran_feat, height):
        """
        Args:
            input (list(torch.tensor)):
                imgs:  (B, N, C, H, W)        # N_views = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            depth:  (B*N, D, fH, fW)
            height: (B*N, H, fH, fW)
            tran_feat: (B*N, C, fH, fW)
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
        """
        self.grid_config = {
            'x': [-40, 40, 0.4],
            'y': [-40, 40, 0.4],
            'z': [-1, 5.4, 6.4],
            'depth': [1.0, 45.0, 0.5],  # (45-1)/0.5
        }
        self.create_grid_infos (**self.grid_config)  # self.grid_size:(200,200,1)
        bev_feat, depth_feat = self.view_transform_core (input, depth, tran_feat)  # (B, C, 1, 200, 200)

        height_map = self.height_feature_to_height_map(height,self.height_range)  # (BxN, fH, fW)
        mask_1,mask_2,mask_3 = self.create_mask_3(height_map,h_min=self.mask_range[0],thr1=self.mask_range[1],thr2=self.mask_range[2],h_max=self.mask_range[3])  # (BxN, fH, fW)
        mask_1_expanded = mask_1.unsqueeze(1).expand_as(tran_feat)  # (BxN, C, fH, fW)
        mask_2_expanded = mask_2.unsqueeze(1).expand_as(tran_feat)
        mask_3_expanded = mask_3.unsqueeze(1).expand_as(tran_feat)

        masked_feat_1 = tran_feat * mask_1_expanded # (BxN, C, fH, fW)
        masked_feat_2 = tran_feat * mask_2_expanded
        masked_feat_3 = tran_feat * mask_3_expanded

        # 最底层
        self.grid_config = self.mask_1_grid
        self.create_grid_infos (**self.grid_config)  # self.grid_size:(200,200,1)
        bev_feat_masked_1, _ = self.view_transform_core (input, depth, masked_feat_1)  # (B,C,4,200,200) (low)

        # 中间层
        self.grid_config = self.mask_2_grid
        self.create_grid_infos (**self.grid_config)  # self.grid_size:(200,200,1)
        bev_feat_masked_2, _ = self.view_transform_core (input, depth, masked_feat_2)  # (B,C,4,200,200) (mid)

        # 最高层
        self.grid_config = self.mask_3_grid
        self.create_grid_infos (**self.grid_config)  # self.grid_size:(200,200,1)
        bev_feat_masked_3, _ = self.view_transform_core (input, depth, masked_feat_3)  # (B,C,8,200,200) (low)

        bev_feat_w_z = torch.cat ((bev_feat_masked_1, bev_feat_masked_2, bev_feat_masked_3), dim=2)  # (B,C,16,200,200) (combined)

        # reset grid_config!
        self.grid_config = {
            'x': [-40, 40, 0.4],
            'y': [-40, 40, 0.4],
            'z': [-1, 5.4, 6.4],
            'depth': [1.0, 45.0, 0.5],  # (45-1)/0.5
        }
        self.create_grid_infos (**self.grid_config)  # self.grid_size:(200,200,1)
        # import pdb;pdb.set_trace()
        return bev_feat, bev_feat_w_z, depth_feat, height


    @force_fp32 ()
    def get_depth_and_height_loss(self, gt_depth, gt_height, depth, height):
        """
        Args:
            gt_depth: (B, N, H, W)
            gt_height: (B, N, H, W) LiDAR点投影到camera坐标系下的gt, H=256,W=704
            height: (B*N, H, fH, fW) 经过heightnet下采样后的高度feat
        Returns:
            loss_height
        """
        B, N, H, W = gt_height.shape
        height_labels = self.get_downsampled_gt_height (gt_height)  # (B*N*fH*fW, H)
        depth_labels = self.get_downsampled_gt_depth (gt_depth)  # (B*N*fH*fW, D)

        fg_mask = torch.max (depth_labels, dim=1).values > 0.0  # bool mask

        height_preds = height.permute (0, 2, 3, 1).contiguous ().view (-1, self.H)  # (B*N*fH*fW, H)
        depth_preds = depth.permute (0, 2, 3, 1).contiguous ().view (-1, self.D)  # (B*N*fH*fW, D)
        height_labels = height_labels[fg_mask]
        height_preds = height_preds[fg_mask]

        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        with autocast (enabled=False):
            height_loss = F.binary_cross_entropy (
                height_preds,
                height_labels,
                reduction='none'
            ).sum () / max (1.0, fg_mask.sum ())

        with autocast (enabled=False):
            depth_loss = F.binary_cross_entropy (
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum () / max (1.0, fg_mask.sum ())

        return self.loss_depth_weight * depth_loss, self.loss_height_weight * height_loss


@NECKS.register_module()
class MGHS_Stereo(MGHS_Depth):
    def __init__(self,  **kwargs):
        super(MGHS_Stereo, self).__init__(**kwargs)
        # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                              kwargs['input_size'],
                                              downsample=4)
