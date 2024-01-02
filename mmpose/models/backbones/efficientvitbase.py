# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, constant_init
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .efficientvit_base.models.nn.drop import apply_drop_func
from typing import Tuple,List,Dict

from .efficientvit_base.models.nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    # FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    # ResBlock,
    ResidualBlock,
)
# from efficientvit.models.utils import build_kwargs_from_config

drop_config = dict(
  name='droppath',
  drop_prob=0.05,
  linear_decay=True,)


class EfficientViTBackbone(nn.Module):
    def __init__(
        self,
        width_list: List[int],
        depth_list: List[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)
        apply_drop_func(self.stages, drop_config)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x:torch.Tensor) -> Dict[str, torch.Tensor]:
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        # print(f'output_dict["stage_final"]is{output_dict["stage_final"]}')
        return output_dict

# @MODELS.register_module()
# def efficientvit_backbone_b0(**kwargs) -> EfficientViTBackbone:
#     backbone = EfficientViTBackbone(
#         width_list=[8, 16, 32, 64, 128],
#         depth_list=[1, 2, 2, 2, 2],
#         dim=16,
#         **build_kwargs_from_config(kwargs, EfficientViTBackbone),
#     )
#     return backbone
class EfficientViTBackboneB0(EfficientViTBackbone):
    def __init__(self, **kwargs):
        super().__init__(width_list=[8, 16, 32, 64, 128], depth_list=[1, 2, 2, 2, 2],dim=16,**kwargs)

class EfficientViTBackboneB1(EfficientViTBackbone):
    def __init__(self, **kwargs):
        super().__init__(width_list=[16, 32, 64, 128, 256], depth_list=[1, 2, 3, 3, 4],dim=16,**kwargs)


class EfficientViTBackboneB2(EfficientViTBackbone):
    def __init__(self, **kwargs):
        super().__init__(width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,**kwargs)

class EfficientViTBackboneB3(EfficientViTBackbone):
    def __init__(self, **kwargs):
        super().__init__(width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,**kwargs)
# @MODELS.register_module()
# def efficientvit_backbone_b2(**kwargs) -> EfficientViTBackbone:
#     backbone = EfficientViTBackbone(
#         width_list=[24, 48, 96, 192, 384],
#         depth_list=[1, 3, 4, 4, 6],
#         dim=32,
#         **build_kwargs_from_config(kwargs, EfficientViTBackbone),
#     )
#     return backbone

# @MODELS.register_module()
# def efficientvit_backbone_b3(**kwargs) -> EfficientViTBackbone:
#     backbone = EfficientViTBackbone(
        # width_list=[32, 64, 128, 256, 512],
        # depth_list=[1, 4, 6, 6, 9],
        # dim=32,
#         **build_kwargs_from_config(kwargs, EfficientViTBackbone),
#     )
#     return backbone
