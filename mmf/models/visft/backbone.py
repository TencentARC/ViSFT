# Copyright (c) Facebook, Inc. and its affiliates.

# Mostly copy-pasted from
# https://github.com/facebookresearch/detr/blob/master/models/backbone.py
"""
Backbone modules.
"""
import math
from collections import OrderedDict
from functools import partial
import torch
import torch.nn.functional as F
import torchvision
from mmf.models.visft.misc import NestedTensor
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from .eva_vit_model import EVAVisionTransformer
# from apex.normalization import FusedLayerNorm
import os
import warnings
from .eva_vit_g import VisionTransformer
warnings.filterwarnings("ignore")
class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": 0}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out = OrderedDict()
        for name, x in xs.items():
            mask = F.interpolate(
                tensor_list.mask[None].float(), size=x.shape[-2:]
            ).bool()[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    
def build_visft_convnet_backbone(args):
    position_embedding = PositionEmbeddingSine(
        args.encoder_hidden_dim // 2, normalize=True
    )
    train_backbone = args.lr_backbone > 0
    return_interm_layers = False
    if args.backbone in ['resnet50', 'resnet101']:
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation
        )
        backbone.train_backbone = train_backbone
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_channels
    elif args.backbone in ['EVA-CLIP-5B']:
        backbone = EVAVisionTransformer(
            img_size=224,
            patch_size=14,
            num_classes=1000,
            use_mean_pooling=False,
            init_values=None,
            patch_dropout=0.,
            embed_dim=1792,
            depth=64,
            # depth=1,
            num_heads=16,
            mlp_ratio=8.571428571428571,
            qkv_bias=True,
            drop_path_rate=0,
            norm_layer= partial(LayerNorm, eps=1e-6),
            xattn=True,
            rope=False,
            postnorm=True,
            pt_hw_seq_len= 16,   # 224/14
            intp_freq=False,
            naiveswiglu=False,
            subln=False,
            train_backbone=train_backbone,
            grad_checkpointing=args.grad_checkpointing
        )
        pretrainedpath = os.path.join(args.backbone_dir)
        checkpoint = torch.load(pretrainedpath, map_location=torch.device("cpu"))
        _tmp_st_output = backbone.load_state_dict(checkpoint, strict=False)
        del checkpoint
        if not train_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
        print(str(_tmp_st_output))
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_features
    elif args.backbone in ['EVA-CLIP-G']:
        backbone = VisionTransformer(
            img_size=224,
            patch_size=14,
            use_mean_pooling=False,
            embed_dim=1408,
            depth=40,
            num_heads=1408//88,
            mlp_ratio=4.3637,
            qkv_bias=True,
            drop_path_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_checkpoint=args.grad_checkpointing,
            train_backbone=train_backbone,
        )
        pretrainedpath = os.path.join(args.backbone_dir)
        checkpoint = torch.load(pretrainedpath, map_location=torch.device("cpu"))
        _tmp_st_output = backbone.load_state_dict(checkpoint, strict=False)
        del checkpoint
        if not train_backbone:
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
        print(str(_tmp_st_output))
        model = Joiner(backbone, position_embedding)
        model.num_channels = backbone.num_features
    return model
