# Copyright (c) OpenMMLab. All rights reserved.
import os

import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.builder import PIPELINES
from torchvision.transforms.functional import rotate

cnt = 0


def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PointToMultiViewDepthandHeight(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        """
        Args:
            points: (N_points, 3):  3: (u, v, d)
            height: int
            width: int

        Returns:
            depth_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)     # (N_points, 2)  2: (u, v)
        depth = points[:, 2]    # (N_points, )哦
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 获取有效投影点.
        coor, depth = coor[kept1], depth[kept1]    # (N, 2), (N, )
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def points2heightmap(self, points, height, width):
        """
        Args:
            points: (N_points, 4):  4: (u, v, d, h)
            height: int
            width: int

        Returns:
            height_map：(H, W)
        """
        height, width = height // self.downsample, width // self.downsample
        height_map = torch.zeros ((height, width), dtype=torch.float32)
        height_mask = torch.zeros ((height, width), dtype=torch.bool)  # mask 用来区分哪些是赋值为0，哪些是投影为0
        coor = torch.round (points[:, :2] / self.downsample)  # (N_points, 2)  2: (u, v)
        height_values = points[:, 3]  # (N_points, )
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
                coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                        points[:, 2] < self.grid_config['depth'][1]) & (
                        points[:, 2] >= self.grid_config['depth'][0])

        # 假设有两个点 (u1, v1, d1, h1) 和 (u2, v2, d2, h2)，它们的图像坐标 (u1, v1) 和 (u2, v2) 相同，
        # 但深度 d1 和 d2 不同。通过上述排序和去重步骤，只保留深度较小的点（靠近传感器），从而确保高度图中每个像素位置只有一个高度值
        # 获取有效投影点
        coor, height_values = coor[kept1], height_values[kept1]  # (N, 2), (N, )
        # 将每个点的图像坐标(u,v)转换为排序权重ranks
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + points[kept1, 2] / 100.).argsort ()
        coor, height_values, ranks = coor[sort], height_values[sort], ranks[sort]
        kept2 = torch.ones (coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, height_values = coor[kept2], height_values[kept2]
        coor = coor.to (torch.long)
        height_map[coor[:, 1], coor[:, 0]] = height_values
        height_mask[coor[:, 1], coor[:, 0]] = True
        return height_map,height_mask

    def __call__(self, results):
        points_lidar = results['points']
        # imgs: (6,3,256,704)
        # sensor2egos: (6,4,4)
        # ego2globals: (6,4,4)
        # intrins: (6,3,3)

        imgs, sensor2egos, ego2globals, intrins = results['img_inputs'][:4]
        # post_rots: (6,3,3)
        # post_trans: (6,3)
        # bda: (3,3)
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        height_map_list = []
        for cid in range(len(results['cam_names'])):  # names: ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
            cam_name = results['cam_names'][cid]    # CAM_TYPE
            # 猜测liadr和cam不是严格同步的，因此lidar_ego和cam_ego可能会不一致.
            # 因此lidar-->cam的路径不采用:   lidar --> ego --> cam
            # 而是： lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]
            # lidar --> lidar_ego --> global --> cam_ego --> cam
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam) # (4,4)

            # points_cam: 将LiDAR点云转换到相机坐标系下
            points_lidar_h = points_lidar.tensor[:, :3]  # (N_points, 3)
            points_cam = points_lidar_h.matmul(lidar2cam[:3, :3].T) + lidar2cam[:3, 3].unsqueeze(0)  # (N_points, 3) 相机坐标系中的高度是Y坐标！！！
            points_ego = points_lidar_h.matmul(lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3].unsqueeze(0)  # (N_points, 3)
            # points_img: 将LiDAR点云转换到图像坐标系下
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)     # (N_points, 3)  3: (ud, vd, d)

            # d: 相机坐标系的深度信息（Z）
            # h: ego坐标系 Z 轴
            points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3], points_ego[:, 2:3]],1)      # (N_points, 4):  4: (u, v, d, h)

            # 再考虑图像增广，只对 (u, v, d) 进行变换
            points_img[:,:3] = points_img[:,:3].matmul(post_rots[cid].T) + post_trans[cid:cid + 1, :]      # (N_points, 4):  4: (u, v, d, h)

            depth_map = self.points2depthmap(points_img[:,:3],
                                             imgs.shape[2],     # H
                                             imgs.shape[3]      # W
                                             )
            height_map,height_mask = self.points2heightmap (points_img,
                                                imgs.shape[2],  # H
                                                imgs.shape[3]  # W
                                                )


            depth_map_list.append(depth_map)
            height_map_list.append(height_map)
        depth_map = torch.stack(depth_map_list)
        height_map = torch.stack(height_map_list)
        results['gt_depth'] = depth_map
        results['gt_height'] = height_map

        # TODO: DEBUG 用的，晚点删掉
        # global cnt
        # if cnt < 10:
        #     keys_to_save = ["gt_depth","gt_height","sample_idx","canvas","pts_filename","img_inputs"]
        #     rusults_to_save = {key: results[key] for key in keys_to_save}
        #     import pickle;

        #     filename = f"pipeline_results_{cnt}.pkl"
        #     cnt = cnt + 1
        #     foldername= r'/opt/data/private/test/FlashOCC/DEBUG'
        #     filepath = foldername + "//" + filename
        #     with open (filepath, 'wb') as file:
        #         pickle.dump (rusults_to_save, file)
        #     print(filepath)
        # cnt = cnt + 1
        return results
