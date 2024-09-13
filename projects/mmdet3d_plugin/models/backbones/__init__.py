from mmdet.models.backbones import ResNet
from .resnet import CustomResNet
from .swin import SwinTransformer
from .unet import UNet


__all__ = ['ResNet', 'CustomResNet', 'SwinTransformer','UNet',]
