# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from mmdet3d.models.builder import NECKS


@NECKS.register_module ()
class Identity (BaseModule):
    def __init__(self):
        super (Identity, self).__init__ ()

    def forward(self, inputs):
        return inputs
