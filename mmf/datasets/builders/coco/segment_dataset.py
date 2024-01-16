# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from contextlib import contextmanager
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import gather_tensor_along_batch, object_to_byte_tensor
from torch import nn, Tensor
from functools import wraps
from mmf.datasets.builders.coco.instances import Instances
from mmf.datasets.builders.coco.boxes import Boxes
from mmf.modules.coco_evaler import CocoEvaluator
from pycocotools.coco import COCO

class SegmentCOCODataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "segment_coco"
        super().__init__(name, config, dataset_type, *args, **kwargs)
        self.dataset_name = name

        image_dir = os.getenv("DATA_PATH", '') + self.config.images[self._dataset_type][imdb_file_index]
        self.image_dir = os.path.join(self.config.data_dir, image_dir)
        coco_json = os.getenv("DATA_PATH", '') + self.config.annotations[self._dataset_type][imdb_file_index]
        self.coco_json = os.path.join(self.config.data_dir, coco_json)

        self.coco_dataset = torchvision.datasets.CocoDetection(
            self.image_dir, self.coco_json
        )
        self.postprocessors = {'segm': PostProcessSegm()}
        if dataset_type != 'train':
            self.coco = COCO(annotation_file=self.coco_json)
            self.evaler = CocoEvaluator(coco_gt=self.coco, iou_types=[k for k in ('segm',)])
       
    def __getitem__(self, idx):
        img, target = self.coco_dataset[idx]
        image_id = self.coco_dataset.ids[idx]
        target = {"image_id": image_id, "annotations": target}

        img, target = self.convert_coco_polys_to_mask(
            {"img": img, "target": target, "dataset_type": self._dataset_type}
        )
        transform_out = self.mask2former_image_and_target_processor(
            {"img": img, "target": target, "dataset_type": self._dataset_type}
        )
        img = transform_out["img"]
        target = transform_out["target"]
        rescaled_h = transform_out["rescaled_h"]
        rescaled_w = transform_out["rescaled_w"]

        current_sample = Sample()
        current_sample.image_id = torch.tensor(image_id, dtype=torch.long)
        current_sample.image = img

        current_sample.orig_size = target["orig_size"].clone().detach()
        current_sample.target_size = torch.as_tensor([int(rescaled_h), int(rescaled_w)])
        current_sample.target_final_size = target['size'].clone().detach()

        current_sample.gt_mask = target['masks'].clone().detach()
        current_sample.labels = target['labels'].clone().detach()
        return current_sample

    def __len__(self):
        return len(self.coco_dataset)

    def format_for_prediction(self, report):
        # gather segmentation output keys across processes
        # support batch == 1 currently
        assert report.batch_size == 1
        outputs_for_evaler = {
            "pred_logits": gather_tensor_along_batch(report.pred_logits),
            "pred_boxes": None,
            "pred_masks": gather_tensor_along_batch(report.pred_masks),
        }
        orig_target_sizes = gather_tensor_along_batch(report.orig_size)
        target_sizes = gather_tensor_along_batch(report.target_size)
        target_final_sizes = gather_tensor_along_batch(report.target_final_size)
        image_id = gather_tensor_along_batch(report.image_id)
        coco_mapping = {
            0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 13:12, 14:13, 15:14, 16:15, 17:16, 18:17, 19:18,
            20:19, 21:20, 22:21, 23:22, 24:23, 25:24, 27:25, 28:26, 31:27, 32:28, 33:29, 34:30, 35:31, 36:32, 37:33, 38:34
            , 39:35, 40:36, 41:37, 42:38, 43:39, 44:40, 46:41, 47:42, 48:43, 49:44, 50:45, 51:46, 52:47, 53:48, 54:49, 55:50
            , 56:51, 57:52, 58:53, 59:54, 60:55, 61:56, 62:57, 63:58, 64:59, 65:60, 67:61, 70:62, 72:63, 73:64, 74:65, 75:66
            , 76:67, 77:68, 78:69, 79:70, 80:71, 81:72, 82:73, 84:74, 85:75, 86:76, 87:77, 88:78, 89:79, 90:80
        }
        class_mapping = {v: k for k, v in coco_mapping.items()}
        results = []
        results = self.postprocessors['segm'](results, outputs_for_evaler, orig_target_sizes, target_sizes, target_final_sizes)

        for sample in results:
            for pred_idx in range(len(sample['labels'])):
                sample['labels'][pred_idx] = class_mapping[sample['labels'][pred_idx].item()]
        
        res = {int(t): output for t, output in zip([image_id], results)}
        self.evaler.update(res)
        return []


    def on_prediction_end(self, predictions):
        # de-duplicate the predictions (duplication is introduced by DistributedSampler)
        self.evaler.synchronize_between_processes()
        self.evaler.accumulate()
        self.evaler.summarize()

        return self.evaler.coco_eval['segm'].stats[0]
    
    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(80, device=scores.device).unsqueeze(0).repeat(100, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(100, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // 80
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise
        
def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped

def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result

class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes, target_final_sizes):
        assert len(orig_target_sizes) == len(max_target_sizes)
        out_logits = outputs['pred_logits']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        # prob = out_logits.sigmoid()
        # max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)

        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)

        # scores = topk_values
        
        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        results = [{'scores': s, 'labels': l} for s, l in zip(scores, labels)]

        # max_h, max_w = 1216, 1216
        max_h, max_w = target_final_sizes.max(0)[0].tolist()

        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)

        # outputs_masks_reduced = torch.gather(
        #     input=outputs_masks,
        #     dim=1,
        #     index=topk_boxes.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, outputs_masks.shape[2], outputs_masks.shape[3])
        # )

        # outputs_masks_reduced = (outputs_masks_reduced.sigmoid() > self.threshold).cpu()
        outputs_masks_reduced = outputs_masks.cpu()


        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks_reduced, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            # results[i]["masks"] = F.interpolate(
            #     results[i]["masks"].float(), size=tuple(tt.tolist()), mode="nearest"
            # ).byte()
            results[i]["masks"] = F.interpolate(results[i]["masks"].float(), size=tuple(tt.tolist()), mode="bilinear").sigmoid() > self.threshold
        return results

