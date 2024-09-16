# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet3d.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.builder import build_head,build_neck
from .bevstereo4d import BEVStereo4D
from .bevdet_occ import BEVDetOCC

@DETECTORS.register_module ()
class DHD (BEVDetOCC):
    def __init__(self,
                 upsample=False,
                 img_voxel_encoder0_backbone=None,
                 img_voxel_encoder0_neck=None,
                 img_voxel_encoder1_backbone=None,
                 img_voxel_encoder1_neck=None,
                 img_voxel_encoder2_backbone=None,
                 img_voxel_encoder2_neck=None,
                 mix=None,
                 **kwargs):
        super (DHD, self).__init__ (**kwargs)
        self.img_voxel_encoder0 = builder.build_backbone (img_voxel_encoder0_backbone)
        self.img_voxel_neck0 = builder.build_neck (img_voxel_encoder0_neck)
        self.img_voxel_encoder1 = builder.build_backbone (img_voxel_encoder1_backbone)
        self.img_voxel_neck1 = builder.build_neck (img_voxel_encoder1_neck)
        self.img_voxel_encoder2 = builder.build_backbone (img_voxel_encoder2_backbone)
        self.img_voxel_neck2 = builder.build_neck (img_voxel_encoder2_neck)
        self.mix = build_neck (mix)
        self.upsample = upsample

    def bev_encoder(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_bev_encoder_backbone (x)
        x = self.img_bev_encoder_neck (x)
        if type (x) in [list, tuple]:
            x = x[0]
        return x

    def voxel_encoder0(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_voxel_encoder0 (x)
        x = self.img_voxel_neck0 (x)
        if type (x) in [list, tuple]:
            x = x[0]
        return x

    def voxel_encoder1(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_voxel_encoder1 (x)
        x = self.img_voxel_neck1 (x)
        if type (x) in [list, tuple]:
            x = x[0]
        return x

    def voxel_encoder2(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_voxel_encoder2 (x)
        x = self.img_voxel_neck2 (x)
        if type (x) in [list, tuple]:
            x = x[0]
        return x

    def extract_img_feat(self, img_inputs, img_metas, **kwargs):
        """ Extract features of images.
        img_inputs:
            imgs:  (B, N_views, 3, H, W)
            sensor2egos: (B, N_views, 4, 4)
            ego2globals: (B, N_views, 4, 4)
            intrins:     (B, N_views, 3, 3)
            post_rots:   (B, N_views, 3, 3)
            post_trans:  (B, N_views, 3)
            bda_rot:  (B, 3, 3)
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N, D, fH, fW)
            height: (B*N, H, fH, fW)
        """
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda = self.prepare_inputs(img_inputs)
        x, _ = self.image_encoder(imgs)    # x: (B, N, C, fH, fW)
        mlp_input = self.img_view_transformer.get_mlp_input(
            sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda)  # (B, N_views, 27)
        x_2d, depth, height, mask_1, mask_2, mask_3 = self.img_view_transformer ([x, sensor2keyegos, ego2globals, intrins, post_rots,
                                              post_trans, bda, mlp_input])
        # x_2d,mask_1,mask_2,mask_3: (B, C, Dy, Dx) x_3d(B, C, 16, 200, 200)
        # depth: (B*N, D, fH, fW)

        x_2d = self.bev_encoder (x_2d)
        x_masked_1 = self.voxel_encoder0 (mask_1)
        x_masked_2 = self.voxel_encoder1 (mask_2)
        x_masked_3 = self.voxel_encoder2 (mask_3)

        x_3d = torch.cat ((x_masked_1, x_masked_2, x_masked_3), dim=1)
        return x_2d, x_3d, depth, height

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        # img_feats: list[bev_0,bev_1]
        x_2d, x_3d, depth, height = self.extract_img_feat (img_inputs, img_metas, **kwargs)
        pts_feats = None
        return x_2d, x_3d, pts_feats, depth, height

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # img_feats: List[(B, C, Dz_1, Dy, Dx),(B, C, Dz_2,Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        # height: (B*N_views, H, fH, fW)
        x_2d, x_3d, pts_feats, depth, height = self.extract_feat (points, img_inputs=img_inputs, img_metas=img_metas,
                                                                  **kwargs)
        losses = dict ()
        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)
        gt_height = kwargs['gt_height']  # (B, N, H, W) eg:(B, 6, 256, 704)
        gt_depth = kwargs['gt_depth'] # (B, N, H, W)
        loss_height = self.img_view_transformer.get_height_loss (gt_depth, gt_height,height)
        losses['loss_height'] = loss_height
        loss_occ = self.forward_occ_train ([x_2d, x_3d], voxel_semantics, mask_camera)
        losses.update (loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: [(B, C, 12, Dy, Dx) ,(B, C, 4,Dy, Dx)]
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        combined = torch.cat (img_feats, dim=1)
        outs = self.mix (combined)
        outs = self.occ_head (outs)
        # assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.occ_head.loss (
            outs,  # (B, Dx, Dy, 16, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        img_feats_2d, img_feats_3d, _, _, _ = self.extract_feat (
            points, img_inputs=img, img_metas=img_metas, **kwargs)

        # occ_bev_feature = img_feats[0]
        # if self.upsample:
        #     occ_bev_feature = F.interpolate(occ_bev_feature, scale_factor=2,
        #                                     mode='bilinear', align_corners=True)

        occ_list = self.simple_test_occ ([img_feats_2d, img_feats_3d],
                                         img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        combined = torch.cat (img_feats, dim=1)
        outs = self.mix (combined)
        outs = self.occ_head (outs)
        occ_preds = self.occ_head.get_occ (outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds


@DETECTORS.register_module ()
class DHD_stereo (BEVStereo4D):
    def __init__(self,
                 occ_head=None,
                 upsample=False,
                 img_voxel_encoder0_backbone=None,
                 img_voxel_encoder0_neck=None,
                 img_voxel_encoder1_backbone=None,
                 img_voxel_encoder1_neck=None,
                 img_voxel_encoder2_backbone=None,
                 img_voxel_encoder2_neck=None,
                 pre_process_net_3d=None,
                 mix=None,
                 **kwargs):
        super (DHD_stereo,self).__init__ (**kwargs)
        self.occ_head = build_head (occ_head)
        self.img_voxel_encoder0 = builder.build_backbone (img_voxel_encoder0_backbone)
        self.img_voxel_neck0 = builder.build_neck (img_voxel_encoder0_neck)
        self.img_voxel_encoder1 = builder.build_backbone (img_voxel_encoder1_backbone)
        self.img_voxel_neck1 = builder.build_neck (img_voxel_encoder1_neck)
        self.img_voxel_encoder2 = builder.build_backbone (img_voxel_encoder2_backbone)
        self.img_voxel_neck2 = builder.build_neck (img_voxel_encoder2_neck)
        if self.pre_process:
            self.pre_process_net_3d = builder.build_backbone (pre_process_net_3d)
        self.mix = builder.build_neck (mix)
        self.pts_bbox_head = None
        self.upsample = upsample

    # 添加 voxel_encoder
    def voxel_encoder0(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_voxel_encoder0 (x)
        x = self.img_voxel_neck0 (x)
        if type (x) in [list, tuple]:
            x = x[0]
        return x

    def voxel_encoder1(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_voxel_encoder1 (x)
        x = self.img_voxel_neck1 (x)
        if type (x) in [list, tuple]:
            x = x[0]
        return x

    def voxel_encoder2(self, x):
        """
        Args:
            x: (B, C, Dy, Dx)
        Returns:
            x: (B, C', 2*Dy, 2*Dx)
        """
        x = self.img_voxel_encoder2 (x)
        x = self.img_voxel_neck2 (x)
        if type (x) in [list, tuple]:
            x = x[0]
        return x

    # 重写 prepare_bev_feat  return bev_feat_2d,bev_feat_3d, depth, stereo_feat
    def prepare_bev_feat(self, img, sensor2keyego, ego2global, intrin,
                         post_rot, post_tran, bda, mlp_input, feat_prev_iv,
                         k2s_sensor, extra_ref_frame):
        """
        Args:
            img:  (B, N_views, 3, H, W)
            sensor2keyego: (B, N_views, 4, 4)
            ego2global: (B, N_views, 4, 4)
            intrin: (B, N_views, 3, 3)
            post_rot: (B, N_views, 3, 3)
            post_tran: (B, N_views, 3)
            bda: (B, 3, 3)
            mlp_input: (B, N_views, 27)
            feat_prev_iv: (B*N_views, C_stereo, fH_stereo, fW_stereo) or None
            k2s_sensor: (B, N_views, 4, 4) or None
            extra_ref_frame:
        Returns:
            bev_feat: (B, C, Dy, Dx)
            depth: (B*N, D, fH, fW)
            stereo_feat: (B*N_views, C_stereo, fH_stereo, fW_stereo)
        """
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat (img)  # (B*N_views, C_stereo, fH_stereo, fW_stereo)
            return None, None, None, None,stereo_feat
        # x: (B, N_views, C, fH, fW)
        # stereo_feat: (B*N, C_stereo, fH_stereo, fW_stereo)
        x, stereo_feat = self.image_encoder (img, stereo=True)

        # 建立cost volume 所需的信息.
        metas = dict (k2s_sensor=k2s_sensor,  # (B, N_views, 4, 4)
                      intrins=intrin,  # (B, N_views, 3, 3)
                      post_rots=post_rot,  # (B, N_views, 3, 3)
                      post_trans=post_tran,  # (B, N_views, 3)
                      frustum=self.img_view_transformer.cv_frustum.to (x),  # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                      cv_downsample=4,
                      downsample=self.img_view_transformer.downsample,
                      grid_config=self.img_view_transformer.grid_config,
                      cv_feat_list=[feat_prev_iv, stereo_feat]
                      )
        # bev_feat_2d: (B, C, Dy, Dx)/(B, C, 1, Dy, Dx)
        # bev_feat_3d: (B, C*16, Dy, Dx)/(B, C, 1, Dy, Dx)
        # depth: (B * N, D, fH, fW) D=88
        # height: (B * N, H, fH, fW) H=33
        bev_feat_2d, bev_feat_3d, depth, height = self.img_view_transformer (
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda,
             mlp_input], metas)

        if self.pre_process:
            if len (bev_feat_3d.size ()) == 5:
                bev_feat_2d_collapse_z = torch.cat (bev_feat_2d.unbind (dim=2), 1)
                bev_feat_3d_collapse_z = torch.cat (bev_feat_3d.unbind (dim=2), 1)
                bev_feat_2d = self.pre_process_net (bev_feat_2d_collapse_z)[0]  # (B, C, Dy, Dx)
                bev_feat_3d = self.pre_process_net_3d (bev_feat_3d_collapse_z)[0]  # (B, C*16, Dy, Dx)
                bev_feat_2d_restore = torch.stack (torch.chunk (bev_feat_2d, 1, dim=1), dim=2)
                bev_feat_3d_restore = torch.stack (torch.chunk (bev_feat_3d, 16, dim=1), dim=2)
                return bev_feat_2d_restore, bev_feat_3d_restore, depth, height,stereo_feat
            # else:
            #     bev_feat_2d = self.pre_process_net (bev_feat_2d)[0]  # (B, C, Dy, Dx)
            #     bev_feat_3d = self.pre_process_net_3d (bev_feat_3d)[0]  # (B, C*16, Dy, Dx)


        return bev_feat_2d, bev_feat_3d, depth, height,stereo_feat

    # 重写 extract_img_feat: view transform 需要返回4个值
    def extract_img_feat(self,
                         img_inputs,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        """
        Args:
            img_inputs:
                imgs:  (B, N, 3, H, W)        # N = 6 * (N_history + 1)
                sensor2egos: (B, N, 4, 4)
                ego2globals: (B, N, 4, 4)
                intrins:     (B, N, 3, 3)
                post_rots:   (B, N, 3, 3)
                post_trans:  (B, N, 3)
                bda_rot:  (B, 3, 3)
            img_metas:
            **kwargs:
        Returns:
            x: [(B, C', H', W'), ]
            depth: (B*N_views, D, fH, fW)
        """
        # self.extract_img_feat_sequential 不需要用！
        if sequential:
            return self.extract_img_feat_sequential (img_inputs, kwargs['feat_prev'])

        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
        bda, curr2adjsensor = self.prepare_inputs (img_inputs, stereo=True)
        # imgs: list[ (B, N, 3, 256, 704)x num_frame ]
        """Extract features of images."""
        bev_feat_2d_list = []
        bev_feat_3d_list = []
        depth_key_frame = None
        height_key_frame = None
        feat_prev_iv = None

        for fid in range (self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = \
                imgs[fid], sensor2keyegos[fid], ego2globals[fid], intrins[fid], \
                post_rots[fid], post_trans[fid]
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                mlp_input = self.img_view_transformer.get_mlp_input (
                    sensor2keyegos[0], ego2globals[0], intrin,
                    post_rot, post_tran, bda)  # (B, N_views, 27)

                inputs_curr = (img, sensor2keyego, ego2global, intrin,
                               post_rot, post_tran, bda, mlp_input,
                               feat_prev_iv, curr2adjsensor[fid],
                               extra_ref_frame)

                if key_frame:
                    bev_feat_2d, bev_feat_3d, depth, height,feat_curr_iv = \
                        self.prepare_bev_feat (*inputs_curr)
                    depth_key_frame = depth
                    height_key_frame = height
                else:
                    with torch.no_grad ():
                        bev_feat_2d, bev_feat_3d, depth, height,feat_curr_iv = \
                            self.prepare_bev_feat (*inputs_curr)

                if not extra_ref_frame:
                    bev_feat_2d_list.append (bev_feat_2d)
                    bev_feat_3d_list.append (bev_feat_3d)
                if not key_frame:
                    feat_prev_iv = feat_curr_iv

        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1  # batch_size = 1
            feat_prev_2d = torch.cat (bev_feat_2d_list[1:], dim=0)
            feat_prev_3d = torch.cat (bev_feat_3d_list[1:], dim=0)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            ego2globals_curr = \
                ego2globals[0].repeat (self.num_frame - 2, 1, 1, 1)
            # (1, N_views, 4, 4) --> (N_prev, N_views, 4, 4)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat (self.num_frame - 2, 1, 1, 1)
            ego2globals_prev = torch.cat (ego2globals[1:-1], dim=0)  # (N_prev, N_views, 4, 4)
            sensor2keyegos_prev = torch.cat (sensor2keyegos[1:-1], dim=0)  # (N_prev, N_views, 4, 4)
            bda_curr = bda.repeat (self.num_frame - 2, 1, 1)  # (N_prev, 3, 3)

            return feat_prev_2d, feat_prev_3d, \
                   [imgs[0],  # (1, N_views, 3, H, W)
                    sensor2keyegos_curr,  # (N_prev, N_views, 4, 4)
                    ego2globals_curr,  # (N_prev, N_views, 4, 4)
                    intrins[0],  # (1, N_views, 3, 3)
                    sensor2keyegos_prev,  # (N_prev, N_views, 4, 4)
                    ego2globals_prev,  # (N_prev, N_views, 4, 4)
                    post_rots[0],  # (1, N_views, 3, 3)
                    post_trans[0],  # (1, N_views, 3, )
                    bda_curr,  # (N_prev, 3, 3)
                    feat_prev_iv,
                    curr2adjsensor[0]]

        if not self.with_prev:
            bev_feat_key_2d = bev_feat_2d_list[0]
            bev_feat_key_3d = bev_feat_3d_list[0]
            if len (bev_feat_key_2d.shape) == 4:
                b, c, h, w = bev_feat_key_2d.shape
                bev_feat_2d_list = \
                    [torch.zeros ([b,
                                   c * (self.num_frame -
                                        self.extra_ref_frames - 1),
                                   h, w]).to (bev_feat_key_2d), bev_feat_key_2d]
                bev_feat_3d_list = \
                    [torch.zeros ([b,
                                   c * (self.num_frame -
                                        self.extra_ref_frames - 1),
                                   h, w]).to (bev_feat_key_3d), bev_feat_key_3d]
            else:
                b, c, z, h, w = bev_feat_key_2d.shape
                bev_feat_2d_list = \
                    [torch.zeros ([b,
                                   c * (self.num_frame -
                                        self.extra_ref_frames - 1), z,
                                   h, w]).to (bev_feat_key_2d), bev_feat_key_2d]
                bev_feat_3d_list = \
                    [torch.zeros ([b,
                                   c * (self.num_frame -
                                        self.extra_ref_frames - 1), z,
                                   h, w]).to (bev_feat_key_3d), bev_feat_key_3d]

        if self.align_after_view_transfromation:
            for adj_id in range (self.num_frame - 2):
                bev_feat_2d_list[adj_id] = self.shift_feature (
                    bev_feat_2d_list[adj_id],  # (B, C, Dy, Dx)
                    [sensor2keyegos[0],  # (B, N_views, 4, 4)
                     sensor2keyegos[self.num_frame - 2 - adj_id]],  # (B, N_views, 4, 4)
                    bda  # (B, 3, 3)
                )  # (B, C, Dy, Dx)
                bev_feat_3d_list[adj_id] = self.shift_feature (
                    bev_feat_3d_list[adj_id],  # (B, C, Dy, Dx)
                    [sensor2keyegos[0],  # (B, N_views, 4, 4)
                     sensor2keyegos[self.num_frame - 2 - adj_id]],  # (B, N_views, 4, 4)
                    bda  # (B, 3, 3)
                )  # (B, C, Dy, Dx)
        bev_feat_2d = torch.cat (bev_feat_2d_list, dim=1)  # (B, C*2, 1, 200, 200)
        bev_feat_3d = torch.cat (bev_feat_3d_list, dim=1)  # (B, C*2, 16, 200, 200)

        bev_feat_2d = torch.cat(bev_feat_2d.unbind (dim=2), 1)  # # (B, C, Dz, Dy, Dx) - > (B, C*Dz, Dy, Dx)

        # 对bev_feat_3d整体voxel切片处理
        x_3d_0 = bev_feat_3d[:, :, :4, :, :]  # (B, C, Dz=4, Dy, Dx)
        x_3d_1 = bev_feat_3d[:, :, 4:8, :, :]  # (B, C, Dz=4, Dy, Dx)
        x_3d_2 = bev_feat_3d[:, :, 8:, :, :]  # (B, C, Dz=8, Dy, Dx)
        x_3d_0_colz = torch.cat (x_3d_0.unbind (dim=2), 1)  # (B, C*Dz=4, Dy, Dx)
        x_3d_1_colz = torch.cat (x_3d_1.unbind (dim=2), 1)  # (B, C*Dz=4, Dy, Dx)
        x_3d_2_colz = torch.cat (x_3d_2.unbind (dim=2), 1)  # (B, C*Dz=8, Dy, Dx)

        # 处理投影到(200x200x1)网格中的feature
        x_2d = self.bev_encoder (bev_feat_2d)

        # 处理投影到(200x200x16)网格中的feature
        x_3d_0_bev = self.voxel_encoder0 (x_3d_0_colz)
        x_3d_1_bev = self.voxel_encoder1 (x_3d_1_colz)
        x_3d_2_bev = self.voxel_encoder2 (x_3d_2_colz)
        x_3d = torch.cat ((x_3d_0_bev, x_3d_1_bev, x_3d_2_bev), dim=1) # (B, C_out, Dy, Dx)

        return x_2d, x_3d, depth_key_frame,height_key_frame

    def extract_feat(self, points, img_inputs, img_metas, **kwargs):
        """Extract features from images and points."""
        """
        points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
        img_inputs:
                imgs:  (B, N_views, 3, H, W)        
                sensor2egos: (B, N_views, 4, 4)
                ego2globals: (B, N_views, 4, 4)
                intrins:     (B, N_views, 3, 3)
                post_rots:   (B, N_views, 3, 3)
                post_trans:  (B, N_views, 3)
                bda_rot:  (B, 3, 3)
        """
        x_2d, x_3d, depth,height = self.extract_img_feat (img_inputs, img_metas, **kwargs)
        pts_feats = None
        return x_2d, x_3d, pts_feats, depth,height

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample. dict_keys(['box_mode_3d', 'box_type_3d', 'sample_idx', 'pts_filename'])
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        # kwargs: dict_keys(['gt_depth', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # x_2d: (B,C_1,200,200)
        # x_3d: (B,C_2,200,200)
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        # height:(B*N_views, H, fH, fW)
        x_2d, x_3d, pts_feats, depth, height = self.extract_feat (
            points, img_inputs=img_inputs, img_metas=img_metas, **kwargs)

        gt_depth = kwargs['gt_depth']   # (B, N_views=6, img_H=256, img_W=704)
        gt_height = kwargs['gt_height']  # (B, N_views=6, img_H=256, img_W=704)

        losses = dict ()
        loss_depth,loss_height = self.img_view_transformer.get_depth_and_height_loss (gt_depth, gt_height,depth,height)
        losses['loss_depth'] = loss_depth
        losses['loss_height'] = loss_height
        voxel_semantics = kwargs['voxel_semantics']  # (B, Dx, Dy, Dz)
        mask_camera = kwargs['mask_camera']  # (B, Dx, Dy, Dz)
        loss_occ = self.forward_occ_train ([x_2d, x_3d], voxel_semantics, mask_camera)
        losses.update (loss_occ)
        return losses

    def forward_occ_train(self, img_feats, voxel_semantics, mask_camera):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            voxel_semantics: (B, Dx, Dy, Dz)
            mask_camera: (B, Dx, Dy, Dz)
        Returns:
        """
        combined = torch.cat (img_feats, dim=1)
        outs = self.mix (combined)
        outs = self.occ_head (outs)
        assert voxel_semantics.min () >= 0 and voxel_semantics.max () <= 17
        loss_occ = self.occ_head.loss (
            outs,  # (B, Dx, Dy, Dz, n_cls)
            voxel_semantics,  # (B, Dx, Dy, Dz)
            mask_camera,  # (B, Dx, Dy, Dz)
        )
        return loss_occ

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        # img_feats: List[(B, C, Dz, Dy, Dx)/(B, C, Dy, Dx) , ]
        # pts_feats: None
        # depth: (B*N_views, D, fH, fW)
        # x_2d, x_3d, pts_feats, depth
        img_feats_2d, img_feats_3d, _, _, _ = self.extract_feat (points, img_inputs=img, img_metas=img_metas, **kwargs)
        occ_list = self.simple_test_occ ([img_feats_2d, img_feats_3d],
                                         img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_list

    def simple_test_occ(self, img_feats, img_metas=None):
        """
        Args:
            img_feats: (B, C, Dz, Dy, Dx) / (B, C, Dy, Dx)
            img_metas:

        Returns:
            occ_preds: List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        """
        combined = torch.cat (img_feats, dim=1)
        outs = self.mix (combined)
        outs = self.occ_head (outs)
        occ_preds = self.occ_head.get_occ (outs, img_metas)  # List[(Dx, Dy, Dz), (Dx, Dy, Dz), ...]
        return occ_preds
