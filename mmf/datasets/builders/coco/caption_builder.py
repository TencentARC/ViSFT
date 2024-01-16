# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.coco.caption_dataset import CaptionCOCODataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("caption_coco")
class CaptionCOCOBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="caption_coco", dataset_class=CaptionCOCODataset
        )

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco/caption.yaml"

