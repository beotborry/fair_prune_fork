from cmath import inf
from syslog import LOG_INFO
from typing import KeysView
import numpy as np
import pandas as pd

import torch.utils.data as data
import data_handler

import pickle

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
        dataset_train, dataset_valid = dataset_train.split([0.8], shuffle=True, seed=0)

        if self.split == 'train':
            del(dataset_test)
            del(dataset_valid)
        elif self.split == 'valid':
            del(dataset_test)

        # for leave out training

        if self.split == 'train' and kwargs['influence_removing'] == 1:

            loss_arr = kwargs['loss_info_filename']

            if "eopp" in kwargs['influence_filename']:
                g0_filename = kwargs['influence_filename'] + "_group0.txt"
                g1_filename = kwargs['influence_filename'] + "_group1.txt"

                with open(g0_filename, "rb") as fp:
                    influence_score_g0 = pickle.load(fp)
                with open(g1_filename, "rb") as fp:
                    influence_score_g1 = pickle.load(fp)

                pp_idx = np.intersect1d(np.where(influence_score_g0 > 0)[0], np.where(influence_score_g1 > 0)[0])
                pn_idx = np.intersect1d(np.where(influence_score_g0 > 0)[0], np.where(influence_score_g1 < 0)[0]) if loss_arr[0] > loss_arr[1] else np.intersect1d(np.where(influence_score_g0 < 0)[0], np.where(influence_score_g1 > 0)[0])
                remainder = np.setdiff1d(np.arange(0, len(dataset_train.features)), np.concatenate((pp_idx, pn_idx)))


                influence_diff = influence_score_g0 - influence_score_g1 if loss_arr[0] > loss_arr[1] else influence_score_g1 - influence_score_g0
                
                influence_diff[remainder] = -float("inf")

                remove_num = int(kwargs['k'] * 0.01 * len(dataset_train.features))
                remove_idx = np.argpartition(influence_diff, -remove_num)[-remove_num:]
                dataset_train = dataset_train.subset(np.setdiff1d(np.arange(0, len(dataset_train.features)), remove_idx))

            elif "eo" in kwargs['influence_filename']:
                g0l0_filename = kwargs['influence_filename'] + "_group0_label0.txt"
                g0l1_filename = kwargs['influence_filename'] + "_group0_label1.txt"
                g1l0_filename = kwargs['influence_filename'] + "_group1_label0.txt"
                g1l1_filename = kwargs['influence_filename'] + "_group1_label1.txt"

                with open(g0l0_filename, "rb") as fp:
                    influence_score_g0l0 = pickle.load(fp)
                with open(g0l1_filename, "rb") as fp:
                    influence_score_g0l1 = pickle.load(fp)
                with open(g1l0_filename, "rb") as fp:
                    influence_score_g1l0 = pickle.load(fp)
                with open(g1l1_filename, "rb") as fp:
                    influence_socre_g1l1 = pickle.load(fp)

                l0_pp_idx = np.intersect1d(np.where(influence_score_g0l0 > 0)[0], np.where(influence_score_g1l0 > 0)[0])
                l0_pn_idx = np.intersect1d(np.where(influence_score_g0l0 > 0)[0], np.where(influence_score_g1l0 < 0)[0]) if loss_arr[0] > loss_arr[1] else np.intersect1d(np.where(influence_score_g0 < 0)[0], np.where(influence_score_g1 > 0)[0])


                
                




            


            # with open(kwargs['influence_filename'], "rb") as fp:
            #     influence_score = pickle.load(fp)
            #     train_idx = np.arange(0, len(dataset_train.features))
            #     remove_num = int(kwargs['k'] * 0.01 * len(dataset_train.features))
            #     remove_idx = np.argpartition(influence_score, -remove_num)[-remove_num:]

            #     train_idx = np.delete(train_idx, remove_idx)
            #     dataset_train = dataset_train.subset(train_idx)

        #if  self.split == 'train' and len(kwargs['influence_scores']) != 0:
        #    train_idx = np.arange(0, len(dataset_train.features))
        #    train_idx = np.delete(train_idx, kwargs['influence_scores'])
        #    dataset_train = dataset_train.subset(train_idx)

        #    print("Removed {} data from train dataset!".format(len(kwargs['influence_scores'])))

        # features, labels = self._balance_test_set(dataset)
        # self.dataset = dataset_train if self.split == 'train' else dataset_test
        if self.split == 'train': self.dataset = dataset_train
        elif self.split == 'valid': self.dataset = dataset_valid
        else: self.dataset = dataset_test
        
        # print(self.dataset.features[0], self.dataset.features[1])
        # print(self.dataset.labels[0], self.dataset.labels[1])
        # print(self.dataset.feature_names)
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
