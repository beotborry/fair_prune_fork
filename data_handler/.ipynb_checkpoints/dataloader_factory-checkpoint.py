from data_handler.dataset_factory import DatasetFactory

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_handler.get_mean_std import get_mean_std
class DataloaderFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataloader(name, batch_size=128, seed=0, num_workers=4,
                       target='Attractive', labelwise=False, group_mode=-1, drop_last=True, sen_attr='sex', skew_ratio=0.8,
                       alpha=None, target_fairness=None, influence_scores=None):

        test_dataset = DatasetFactory.get_dataset(name, split='test', target=target,
                                                  group_mode=group_mode, sen_attr=sen_attr, skew_ratio=skew_ratio, influence_scores=influence_scores)
        train_dataset = DatasetFactory.get_dataset(name, split='train', target=target,
                                                   group_mode=group_mode, sen_attr=sen_attr, skew_ratio=skew_ratio, influence_scores=influence_scores)
        valid_dataset = DatasetFactory.get_dataset(name, split='valid', target=target,
                                                   group_mode=group_mode, sen_attr=sen_attr, skew_ratio=skew_ratio, influence_scores=influence_scores)

        print('# data of test ',  len(test_dataset))
        print('# data of train ', len(train_dataset))
        print('# data of valid ', len(valid_dataset))

        def _init_fn(worker_id):
            np.random.seed(int(seed))
            
        shuffle = True
        sampler = None
        if labelwise:
            if group_mode>0:
                raise ValueError("Group should be -1 if you want to use labelwise")
            from data_handler.custom_loader import Customsampler
            sampler = Customsampler(train_dataset, replacement=False, batch_size=batch_size)
            shuffle = False
            
        elif alpha is not None:         
            # if we use the sampler proposed in FairBatch, alpha is used to control the size of gradient
            from data_handler.fairbatch import FairBatch
            sampler = FairBatch(train_dataset, batch_size, alpha=alpha, target_fairness=target_fairness, seed=seed)
            shuffle = False
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                      shuffle=shuffle, num_workers=num_workers, worker_init_fn=_init_fn,
                                      pin_memory=True, drop_last=False)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, sampler=sampler,
                                      shuffle=shuffle, num_workers=num_workers, worker_init_fn=_init_fn,
                                      pin_memory=True, drop_last=False)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, worker_init_fn=_init_fn, pin_memory=True, drop_last=False)

        print('Dataset loaded.')
        
        num_classes = test_dataset.num_classes
        num_groups = test_dataset.num_groups

        return num_classes, num_groups, train_dataloader, valid_dataloader, test_dataloader

