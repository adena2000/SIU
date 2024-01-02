from typing import List,Dict
import torch.nn as nn
from mmpose.models.backbones.efficientvit.models.efficientvit import (
    EfficientViTSeg,
    efficientvit_seg_b0,
    efficientvit_seg_b1,
    efficientvit_seg_b2,
    efficientvit_seg_b3,
)
from mmpose.models.backbones.efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_seg_model"]


REGISTERED_SEG_MODEL: Dict[str, Dict[str, str]] = {
    "cityscapes": {
        "b0": "/home/s06007/mmpose/checkpoints/b1.pt",
        "b1": "outputs/efficient_Cityscapes_B1_25e4_ori/best_coco-wholebody_AP_epoch_410.pth",
        "b2": "assets/checkpoints/seg/cityscapes/b2.pt",
        "b3": "assets/checkpoints/seg/cityscapes/b3.pt",
    },
    "ade20k": {
        "b1": "assets/checkpoints/seg/ade20k/b1.pt",
        "b2": "assets/checkpoints/seg/ade20k/b2.pt",
        "b3": "assets/checkpoints/seg/ade20k/b3.pt",
    },
}


def create_seg_model(
    name: str, dataset: str, pretrained=True, weight_url: str or None = None, **kwargs
) -> EfficientViTSeg:
    model_dict = {
        "b0": efficientvit_seg_b0,
        "b1": efficientvit_seg_b1,
        "b2": efficientvit_seg_b2,
        "b3": efficientvit_seg_b3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](dataset=dataset, **kwargs)

    if pretrained:
        weight_url = weight_url or REGISTERED_SEG_MODEL[dataset].get(name, None)
        if weight_url is None:
            raise ValueError(f"Do not find the pretrained weight of {name}.")
        else:
            weight = load_state_dict_from_file(weight_url)
            model.load_state_dict(weight)
    return model
