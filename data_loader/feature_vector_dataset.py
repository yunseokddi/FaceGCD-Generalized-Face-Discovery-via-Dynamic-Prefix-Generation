import torch
from torch.utils.data import Dataset

import os
from copy import deepcopy

from .data_utils import MergedDataset


class FeatureVectorDataset(Dataset):

    def __init__(self, base_dataset, feature_root):
        self.base_dataset = base_dataset
        self.target_transform = deepcopy(base_dataset.target_transform)

        self.base_dataset.target_transform = None
        self.base_dataset.transform = None

        self.feature_root = feature_root

        if isinstance(self.base_dataset, MergedDataset):
            self.base_dataset.labelled_dataset.target_transform = None
            self.base_dataset.unlabelled_dataset.target_transform = None

    def __getitem__(self, item):

        if isinstance(self.base_dataset, MergedDataset):

            _, label, uq_idx, mask_lab = self.base_dataset[item]

            feat_path = os.path.join(self.feature_root, f'{label}', f'{uq_idx}.npy')
            feature_vector = torch.load(feat_path)


            if self.target_transform is not None:
                label = self.target_transform(label)

            return feature_vector, label, uq_idx, mask_lab[0]

        else:

            _, label, uq_idx = self.base_dataset[item]

            feat_path = os.path.join(self.feature_root, f'{label}', f'{uq_idx}.npy')

            feature_vector = torch.load(feat_path)


            if self.target_transform is not None:
                label = self.target_transform(label)

            return feature_vector, label, uq_idx


    def __len__(self):
        return len(self.base_dataset)