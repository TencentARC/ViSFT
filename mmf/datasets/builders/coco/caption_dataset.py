# Copyright (c) Facebook, Inc. and its affiliates.
import os
from typing import Dict

import torch
import torch.nn.functional as F
import torchvision
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import gather_tensor_along_batch, object_to_byte_tensor
from torch import nn, Tensor
import torch
import h5py
import json
import os
import torchvision.transforms as transforms

class CaptionCOCODataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        name = "caption_coco"
        super().__init__(name, config, dataset_type, *args, **kwargs)
        self.dataset_name = name
        self.split = self._dataset_type
        # Open hdf5 file where images are stored
        data_folder = self.config.images
        self.data_folder = data_folder
        data_name = 'coco_5_cap_per_img_5_min_word_freq' 
        self.data_name = data_name
        self.imgs = None
        # Load encoded captions (completely into memory)
        with open(os.path.join(os.getenv("DATA_PATH", '') + data_folder, self.split.upper() + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(os.getenv("DATA_PATH", '') + data_folder, self.split.upper() + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform=transforms.Compose([normalize])
        
        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)


    def __getitem__(self, i):
        if self.imgs is None:
            self.h = h5py.File(os.path.join(os.getenv("DATA_PATH", '') + self.data_folder, self.split.upper() + '_IMAGES_' + self.data_name + '.hdf5'), 'r')
            self.imgs = self.h['images']

            # Captions per image
            self.cpi = self.h.attrs['captions_per_image']
            
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        current_sample = Sample()

        if self.split == 'train':
            current_sample.image = img
            current_sample.caption = caption
            current_sample.caplen = caplen
            return current_sample
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            current_sample.image = img
            current_sample.caption = caption
            current_sample.caplen = caplen
            current_sample.all_captions = all_captions
            # return img, caption, caplen, all_captions
            return current_sample


    def __len__(self):
        return self.dataset_size

    def format_for_prediction(self, report):
        hypotheses = list()  # hypotheses (predictions)
        references = list() # references (true captions) for calculating BLEU-4 score

        # gather decoder output keys across processes
        scores_copy = gather_tensor_along_batch(report.scores_copy)
        decode_lengths = gather_tensor_along_batch(report.decode_lengths)
        allcaps = gather_tensor_along_batch(report.all_captions)
        sort_ind = gather_tensor_along_batch(report.sort_ind)

        # References
        # Read word map
        word_map_file = os.path.join(os.getenv("DATA_PATH", '') + '/processed_datasets/coco_caption_hdf5_files/', 'WORDMAP_' + 'coco_5_cap_per_img_5_min_word_freq' + '.json')
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
        allcaps = allcaps[sort_ind]
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
        
        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)

        return hypotheses, references

    



