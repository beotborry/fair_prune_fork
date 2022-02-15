from typing import KeysView
import numpy as np
import pandas as pd

import torch.utils.data as data
import data_handler

# class TabularDataset(data.Dataset):
class TabularDataset(data_handler.GenericDataset):
    """Adult dataset."""
    # 1 idx -> sensi
    # 2 idx -> label
    # 3 idx -> filename or feature (image / tabular)
#     def __init__(self, dataset, sen_attr_idx, normalize=True, root, split='train', labelwise=False, group_mode=-1):
    def __init__(self, dataset, sen_attr_idx, **kwargs):
        super(TabularDataset, self).__init__(**kwargs)
        self.sen_attr_idx = sen_attr_idx
        
        dataset_train, dataset_test = dataset.split([0.8], shuffle=True, seed=0)
        # dataset_train, dataset_valid = dataset_train.split([0.8], shuffle=True, seed=0)
        
        # for leave out training
        # if  self.split == 'train' and len(kwargs['influence_scores']) != 0:
        #     train_idx = np.arange(0, len(dataset_train.features))
        #     train_idx = np.delete(train_idx, kwargs['influence_scores'])
        #     dataset_train = dataset_train.subset(train_idx)

        #     print("Removed {} data from train dataset!".format(len(kwargs['influence_scores'])))

        # features, labels = self._balance_test_set(dataset)
        # self.dataset = dataset_train if self.split == 'train' else dataset_test
        if self.split == 'train': self.dataset = dataset_train
        # elif self.split == 'valid': self.dataset = dataset_valid
        else: self.dataset = dataset_test
        
        features = np.delete(self.dataset.features, self.sen_attr_idx, axis=1)
        mean, std = self._get_mean_n_std(dataset_train.features, self.group_mode)        
        features = (features - mean) / std
        
        self.groups = np.expand_dims(self.dataset.features[:, self.sen_attr_idx], axis=1)
        self.labels = np.squeeze(self.dataset.labels)
        
        # self.features = self.dataset.features
        self.features = np.concatenate((self.groups, self.dataset.labels, features), axis=1)

        # For prepare mean and std from the train dataset
        self.num_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)

    def __len__(self):
        if self.group_mode == -1:
            return np.sum(self.num_data)
        else:
            return np.sum(self.num_data[self.group_mode, :])

    def get_dim(self):
        return self.dataset.features.shape[-1]

    def __getitem__(self, idx):
        if self.group_mode != -1:
            idx = self.gp_idx[self.group_mode][idx]

        features = self.features[idx]
        group = features[0]
        label = features[1]
        #feature = features[2:]
        feature = np.append(features[2:], group)

#         if self.transform:
#             feature = self.transform(feature)
        return np.float32(feature), 0, group, np.int64(label), (idx, 0)

    def _get_mean_n_std(self, train_features, group):
        features = np.delete(train_features, self.sen_attr_idx, axis=1)
        # features = train_features
        if not group == -1:
            features = features[train_features[:, self.sen_attr_idx] == group]
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] += 1e-7
        return mean, std

#     def _make_data(self, dataset):
#         min_cnt = 500
#         data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)

#         split_index_sets = []
#         for idx, feature in enumerate(dataset.features):
#             g, l = int(feature[self.sen_attr_idx]), int(dataset.labels[idx].item())
#             if data_count[g, l] < min_cnt:
#                 split_index_sets.append(idx)
#             data_count[g, l] += 1

#         features = {}
#         labels = {}
#         split_index_sets = np.array(split_index_sets)
#         features['train'] = np.delete(dataset.features, split_index_sets, axis=0)
#         features['test'] = dataset.features[split_index_sets]
#         labels['train'] = np.delete(dataset.labels, split_index_sets, axis=0)
#         labels['test'] = dataset.labels[split_index_sets]

#         return features, labels
