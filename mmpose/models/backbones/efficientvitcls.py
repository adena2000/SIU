# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from .efficientvitbase import EfficientViTBackboneB1,EfficientViTBackboneB2,EfficientViTBackboneB3,EfficientViTBackboneB0
from .efficientvitbase import EfficientViTBackbone
from .efficientvit_base.models.nn import ConvLayer, LinearLayer, OpSequential
from typing import Tuple,List,Dict
from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
import logging
# class ClsHead(OpSequential):
#     def __init__(
#         self,
#         in_channels: int,
#         width_list: List[int],
#         n_classes=1000,
#         dropout=0.0,
#         norm="bn2d",
#         act_func="hswish",
#         fid="stage_final",
#     ):
#         ops = [
#             ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
#             nn.AdaptiveAvgPool2d(output_size=1),
#             LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
#             LinearLayer(width_list[1], n_classes, True, dropout, None, None),
#         ]
#         super().__init__(ops)

#         self.fid = fid

    # def forward(self, feed_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     x = feed_dict[self.fid]
    #     return OpSequential.forward(self, x)

# class EfficientViTCls(nn.Module):
#     def __init__(self, backbone: EfficientViTBackbone, head: ClsHead) -> None:
#         super().__init__()
#         self.backbone = backbone
#         self.head = head

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         feed_dict = self.backbone(x)
#         output = self.head(feed_dict)
#         return output

@MODELS.register_module()  
class Efficientvit_Cls_B0(BaseBackbone):
    # self.backbone = efficientvit_backbone_b1(**kwargs)
    def __init__(self) -> None:
        super().__init__()
        self.backbone=EfficientViTBackboneB0()

    def forward(self,x):
        temp=self.backbone(x)
        output=temp["stage_final"]
        # print(f'output is {output.shape}')
        return output
    
@MODELS.register_module()  
class Efficientvit_Cls_B1(BaseBackbone):
    # self.backbone = efficientvit_backbone_b1(**kwargs)
    def __init__(self) -> None:
        super().__init__()
        self.backbone=EfficientViTBackboneB1()

    def forward(self,x):
        # B, C, H, W = x.shape
        # print(f'backbone is {self.backbone}')
        # print(f'x.shape is {x.shape}')
        # log=logging.getLogger('backbone')
        # log.setLevel('INFO')
        # # 设置输出渠道(以文件方式输出需设置文件路径)
        # file_handler = logging.FileHandler('/home/s06007/mmpose/outputs/efficient_288_1e3/test.log', encoding='utf-8')
        # file_handler.setLevel('INFO')
        # # 设置输出格式(实例化渠道)
        # fmt_str = '%(asctime)s %(thread)d %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        # formatter = logging.Formatter(fmt_str)
        # # 绑定渠道的输出格式
        # file_handler.setFormatter(formatter)
        # # 绑定渠道到日志收集器
        # log.addHandler(file_handler)
        # log.info(self.backbone)

        temp=self.backbone(x)
        output=temp["stage_final"]
        # print(f'output is {output.shape}')
        return output
    
@MODELS.register_module()  
class Efficientvit_Cls_B2(BaseBackbone):
    # self.backbone = efficientvit_backbone_b1(**kwargs)
    def __init__(self) -> None:
        super().__init__()
        self.backbone=EfficientViTBackboneB2()

    def forward(self,x):
        temp=self.backbone(x)
        output=temp["stage_final"]
        # print(f'output is {output.shape}')
        return output
    
@MODELS.register_module()  
class Efficientvit_Cls_B3(BaseBackbone):
    # self.backbone = efficientvit_backbone_b1(**kwargs)
    def __init__(self) -> None:
        super().__init__()
        self.backbone=EfficientViTBackboneB3()

    def forward(self,x):
        temp=self.backbone(x)
        output=temp["stage_final"]
        # print(f'output is {output.shape}')
        return output