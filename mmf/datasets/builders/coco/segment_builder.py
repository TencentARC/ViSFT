# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmf.common.registry import registry
from mmf.datasets.builders.coco.segment_dataset import SegmentCOCODataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("segment_coco")
class SegmentCOCOBuilder(MMFDatasetBuilder):
    def __init__(self):
        super().__init__(
            dataset_name="segment_coco", dataset_class=SegmentCOCODataset
        )

    @classmethod
    def config_path(cls):
        return "configs/datasets/coco/segment.yaml"

