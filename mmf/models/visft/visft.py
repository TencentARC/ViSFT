# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from copy import deepcopy
from typing import Dict

import torch.nn.functional as F
import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.models.visft.visft_heads import (
    AttributeHead,
    build_detection_loss,
    MLP,
    ViSFTHeads,
)
from mmf.modules.encoders import TransformerEncoder
from mmf.utils.distributed import byte_tensor_to_object
from torch import nn, Tensor
from mmf.models.visft.segment_criterion import SetCriterion
from mmf.models.visft.segment_matcher import HungarianMatcher
try:
    from transformers3.modeling_bert import BertPredictionHeadTransform
except ImportError:
    from transformers.modeling_bert import BertPredictionHeadTransform

from mmf.models.visft.caption_decoder import accuracy
import loralib as lora
logger = logging.getLogger(__name__)


@registry.register_model("visft")
class ViSFT(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    @classmethod
    def config_path(cls):
        return "configs/models/visft/defaults.yaml"


    def build(self):
        # build the base model (based on DETR)
        self.visft_heads = ViSFTHeads(self.config.base_args)

        def keep_only_backbone_params(model_state_dict):
            keys = list(model_state_dict.keys())
            for k in keys:
                if "backbone" not in k:
                    model_state_dict.pop(k)

        segment_ckpt_path = self.config.segment_ckpt_path
        detection_ckpt_path = self.config.detection_ckpt_path
        caption_ckpt_path = self.config.caption_ckpt_path
            
        # build the text encoder (BERT)
        # self.bert_model = TransformerEncoder(self.config.base_args.bert_config)
        detr_hidden_dim = self.config.base_args.decoder_hidden_dim

        for dataset_name in self.config.base_args.num_queries.get("detection", []):
            num_cls = self.config.heads["detection"][dataset_name]["num_classes"]
            self.visft_heads.class_embeds = nn.Linear(detr_hidden_dim, num_cls + 1)
            self.visft_heads.bbox_embeds = MLP(detr_hidden_dim, detr_hidden_dim, 4, 3)
            attr_head = None
            self.det_losses = build_detection_loss(
                self.config.base_args, num_cls, attr_head
            )

        # stage2 initialization
        if segment_ckpt_path != "" or detection_ckpt_path != "" or caption_ckpt_path != "":
            logger.info(f"initializing segment head params from {segment_ckpt_path}")
            segment_head_ckpt = torch.load(segment_ckpt_path, map_location=torch.device("cpu"))

            logger.info(f"initializing detection head params from {detection_ckpt_path}")
            detection_head_ckpt = torch.load(detection_ckpt_path, map_location=torch.device("cpu"))

            logger.info(f"initializing caption head params from {caption_ckpt_path}")
            caption_head_ckpt = torch.load(caption_ckpt_path, map_location=torch.device("cpu"))

            head_ckpt = {}
            for key, value in list(segment_head_ckpt.items()):
                if 'segment' in key:
                    head_ckpt[key] = value
            
            for key, value in list(caption_head_ckpt.items()):
                if 'caption' in key:
                    head_ckpt[key] = value 

            for key, value in list(detection_head_ckpt.items()):
                if 'segment' not in key and 'caption' not in key:
                     head_ckpt[key] = value
            
            del segment_head_ckpt
            del caption_head_ckpt
            del detection_head_ckpt

            _tmp_load_output = self.load_state_dict(
                    head_ckpt, strict=False
                 )
            print(str(_tmp_load_output))
        self.loss_calculation_fn = {}
        self.loss_calculation_fn["detection"] = self.detection_loss_calculation
        self.loss_calculation_fn["caption"] = self.caption_loss_calculation
        weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0}
        aux_weight_dict = {}
        for i in range(9):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        self.loss_calculation_fn["segment"] = SetCriterion(
            num_classes=80,
            matcher=HungarianMatcher(
                cost_class=2.0,
                cost_mask=5.0,
                cost_dice=5.0,
                num_points=12544,
            ),
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=["labels", "masks"],
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
        )

        self.losses_dict = {}
        self.losses_dict["caption"] = {
            name: self.get_loss_fn(self.config.heads["caption"][name]["loss_type"])
            for name in self.config.heads["caption"]
        }

    def forward_bert_with_task_idx(self, sample_list):
        bert = self.bert_model.module
        input_ids = sample_list.input_ids
        attention_mask = sample_list.input_mask
        token_type_ids = sample_list.segment_ids
        device = input_ids.device

        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=device)

        input_shape = input_ids.size()

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        start_idx = 0
        if self.config.base_args.use_task_embedding_in_lang_encoder:
            bs = input_ids.size(0)
            task_idx = self.get_task_idx(sample_list.dataset_name)
            task_embed = self.task_embeddings_lang.weight[task_idx]
            task_embed = task_embed.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1)
            embedding_output = torch.cat([task_embed, embedding_output], dim=1)
            task_attention_mask = embedding_output.new_ones((bs, 1))
            attention_mask = torch.cat([task_attention_mask, attention_mask], dim=1)
            start_idx = 1

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None for _ in range(bert.config.num_hidden_layers)]
        encoder_outputs = bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs[0][:, start_idx:, :]
        pos_embeddings = self.bert_model.embeddings.position_embeddings(position_ids)

        return sequence_output, pos_embeddings

    def forward(self, sample_list):
        detr_outputs = {}
        task_type = self.get_task_type(sample_list.dataset_name)
        text_src = None
        text_mask = None
        text_pos = None
        img_src = None
        caps = None
        caplens = None
 
        img_src = sample_list.image
        if task_type == "caption":
            caps = sample_list.caption
            caplens = sample_list.caplen

        detr_outputs = self.visft_heads(
            img_src=img_src,
            text_src=text_src,
            text_mask=text_mask,
            text_pos=text_pos,
            task_type=task_type,
            dataset_name=sample_list.dataset_name,
            task_idx=self.get_task_idx(sample_list.dataset_name),
            caps=caps,
            caplens=caplens,
        )
        if task_type == "segment" and not self.training:
            # for inference
            detr_outputs['inference_shape'] = img_src.shape[-2:]
            output = detr_outputs
        else:
            output = self.loss_calculation_fn[task_type](detr_outputs, sample_list)
        return output

    def caption_loss_calculation(self, detr_outputs: Dict[str, Tensor], sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        scores = detr_outputs["scores"]
        targets = detr_outputs["targets"]
        alphas = detr_outputs["alphas"]
        losses = {}
        metrics = {}

        if sample_list.dataset_type != "test":
            loss_prefix = f"{sample_list.dataset_type}/{sample_list.dataset_name}/"
            loss = self.losses_dict[task_type][sample_list.dataset_name](
                scores, targets
            )
            # Add doubly stochastic attention regularization
            loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()
            losses[loss_prefix + f"loss"] = loss
            metrics[loss_prefix + f"top5acc"] = accuracy(scores, targets, 5)
        
        detr_outputs["scores"] = scores
        detr_outputs["losses"] = losses
        detr_outputs["metrics"] = metrics
        return detr_outputs


    def detection_loss_calculation(self, detr_outputs: Dict[str, Tensor], sample_list):
        hs = detr_outputs["hidden_states"]

        outputs_class = self.visft_heads.class_embeds(hs)
        outputs_coord = self.visft_heads.bbox_embeds(hs).sigmoid()
        detr_outputs.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "hs_for_attr": hs[-1],
            }
        )
        # skip loss computation on test set (which usually doesn't contain labels)
        if sample_list.dataset_type != "test":
            if self.config.base_args.aux_loss:
                detr_outputs["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b, "hs_for_attr": c}
                    for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], hs[:-1])
                ]

            criterion = self.det_losses
            targets = [byte_tensor_to_object(t) for t in sample_list.targets_enc]
            targets = [{k: v.to(hs.device) for k, v in t.items()} for t in targets]
            sample_list.targets = targets
            loss_dict = criterion(detr_outputs, sample_list.targets)
            weight_dict = criterion.weight_dict
            loss_prefix = f"{sample_list.dataset_type}/{sample_list.dataset_name}/"
            losses = {
                (loss_prefix + f"{k}"): loss_dict[k]
                * weight_dict[k]
                * self.config.detection_loss_weight
                for k in loss_dict.keys()
                if k in weight_dict
            }
            detr_outputs["losses"] = losses

        if (
            self.config.heads["detection"][sample_list.dataset_name]["use_attr"]
            and self.config.predict_attributes
        ):
            hs_for_attr = detr_outputs["hs_for_attr"]
            top_obj_class = detr_outputs["pred_logits"][..., :-1].argmax(dim=-1)
            attr_head = self.det_losses.attribute_head
            detr_outputs["attr_logits"] = attr_head(hs_for_attr, top_obj_class)

        return detr_outputs

    def classifier_loss_calculation(self, detr_outputs: Dict[str, Tensor], sample_list):
        task_type = self.get_task_type(sample_list.dataset_name)
        hs = detr_outputs["hidden_states"]
        if not self.config.loss_on_all_hs:
            hs = detr_outputs["hidden_states"][-1:]
        num_queries = self.config.base_args.num_queries[task_type][
            sample_list.dataset_name
        ]
        assert hs[0].size(1) == num_queries
        losses = {}
        scores = None
        detr_outputs = {}
        num_labels = self.config.heads[task_type][sample_list.dataset_name][
            "num_labels"
        ]

        for idx, current_hs in enumerate(hs):
            pooled_output = current_hs[:, -num_queries, :]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifiers[task_type][sample_list.dataset_name](
                pooled_output
            )
            reshaped_logits = logits.contiguous().view(-1, num_labels)
            scores = reshaped_logits
            # skip loss computation on test set (which usually doesn't contain labels)
            if sample_list.dataset_type != "test":
                loss_prefix = f"{sample_list.dataset_type}/{sample_list.dataset_name}/"
                loss = self.losses_dict[task_type][sample_list.dataset_name](
                    scores, sample_list.targets
                )
                if sample_list.dataset_name == "vqa2":
                    loss *= sample_list.targets.size(1)
                losses[loss_prefix + f"loss_{idx}"] = loss

        detr_outputs["scores"] = scores
        detr_outputs["losses"] = losses
        return detr_outputs

    def get_optimizer_parameters(self, config):
        # stage2 param setting
        if config.model_config['visft'].segment_ckpt_path != "" or config.model_config['visft'].detection_ckpt_path != "" or config.model_config['visft'].caption_ckpt_path != "":
            for n, p in self.visft_heads.named_parameters():
                # grad checkpoint requires patch embed to have grads
                if 'lora' not in n and 'patch_embed' not in n:
                    p.requires_grad = False
            detr_params = [
                {
                    "params": [
                        p
                        for n, p in self.visft_heads.named_parameters()
                        if "backbone" in n and p.requires_grad and 'patch_embed' not in n
                    ],
                    "lr": self.config.base_args.lr_backbone,
                },
            ]
            return detr_params
        # stage1 training param settings
        # no lora in backbone, backbone freeze
        lora.mark_only_lora_as_trainable(self.visft_heads.backbone)
        # in-domain task heads
        detr_params = [
            {
                "params": [
                    p
                    for n, p in self.visft_heads.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            }
        ]

        return detr_params

    def get_task_idx(self, dataset_name):
        task_type = self.get_task_type(dataset_name)
        assert task_type in self.config.heads
        return self.config.heads[task_type][dataset_name]["task_idx"]

    def get_task_type(self, dataset_name):
        task_type = "detection"
        if dataset_name in self.config.heads["caption"]:
            task_type = "caption"
        elif dataset_name in self.config.heads['segment']:
            task_type = 'segment'
        return task_type

    def get_loss_fn(self, loss_type):
        if loss_type == "binary_cross_entropy_with_logits":
            return nn.functional.binary_cross_entropy_with_logits
        elif loss_type == "cross_entropy":
            return nn.functional.cross_entropy
        else:
            raise Exception(f"Unknown loss type: {loss_type}")
