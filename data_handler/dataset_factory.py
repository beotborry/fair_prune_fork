# from data_handler.adv_dataset import AdvDataset
import importlib
import torch.utils.data as data
import numpy as np
from collections import defaultdict
dataset_dict = {'utkface' : ['data_handler.utkface','UTKFaceDataset'],
                'celeba' : ['data_handler.celeba', 'CelebA'],
                'adult' : ['data_handler.adult', 'AdultDataset_torch'],
                'compas' : ['data_handler.compas', 'CompasDataset_torch'],
                'bank' : ['data_handler.bank', 'BankDataset_torch'],
                'cifar10s' : ['data_handler.cifar10s', 'CIFAR10_S'],
                'cifar10cg' : ['data_handler.cifar10s', 'CIFAR10_CG'],
                'credit' : ['data_handler.credit', 'CreditDataset_torch'],
                'retiring_adult': ['data_handler.retiring_adult', 'RetiringDataset_torch'],
                'retiring_adult_coverage' : ['data_handler.retiring_adult_coverage', 'RetiringCoverageDataset_torch']
               }

class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, split='train', target='Attractive',group_mode=-1, sen_attr='sex', skew_ratio=0.8):
        # make kwargs
        root = f'./data/{name}'
        kwargs = {'root':root,
                  'split':split,
                  'group_mode':group_mode
                  }
        tabular_datas = ['adult','compas', 'credit', 'bank', 'retiring_adult', 'retiring_adult_coverage']
        if name in tabular_datas:
            kwargs['sen_attr'] = sen_attr
        if name == 'celeba':
            kwargs['target_attr'] = target
        if name == 'cifar10s':
            kwargs['skewed_ratio'] = skew_ratio

            # call the class
        if name not in dataset_dict.keys():
            raise Exception('Not allowed method')
            
        module = importlib.import_module(dataset_dict[name][0])
        class_ = getattr(module, dataset_dict[name][1])
        return class_(**kwargs)

class GenericDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, group_mode=-1):
        self.root = root
        self.split = split
        self.transform = transform
        self.group_mode = group_mode
        self.num_data = None
        
    def __len__(self):
        if self.group_mode == -1:
            return np.sum(self.num_data)
        else:
            return np.sum(self.num_data[self.group_mode, :])
    
    def _data_count(self, features, num_groups, num_classes):
        idxs_per_group = defaultdict(lambda: [])
        data_count = np.zeros((num_groups, num_classes), dtype=int)

        for idx, i in enumerate(features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            idxs_per_group[(s,l)].append(idx)
            
        print(f'mode : {self.split}')        
        for i in range(num_groups):
            print('# of %d group data : '%i, data_count[i, :])
        return data_count, idxs_per_group
            
    def _make_data(self, features, num_groups, num_classes):
        # if the original dataset not is divided into train / test set, this function is used
        #dataset_size = self.__len__()

        import copy
        min_cnt = 100

        data_count = np.zeros((num_groups, num_classes), dtype=int)
        tmp = []
        for i in reversed(self.features):
            s, l = int(i[0]), int(i[1])
            data_count[s, l] += 1
            if data_count[s, l] <= min_cnt:
                features.remove(i)
                tmp.append(i)
        
        features = features if self.split =='train' else tmp
        return features
        
    
    def _balance_test_data(self, num_data, num_groups, num_classes):
        # if the original dataset is divided into train / test set, this function is used        
        num_data_min = np.min(num_data)
        print('min : ', num_data_min)
        data_count = np.zeros((num_groups, num_classes), dtype=int)
        new_features = []
#         print(len(self.attr))     
        for idx, i in enumerate(self.features):
            s, l = int(i[0]), int(i[1])
            if data_count[s, l] < num_data_min:
                new_features.append(i)
#                 new_filename.append(self.filename[index])
#                 new_attr.append(self.attr[index])
                data_count[s, l] += 1
            
#         for i in range(self.num_groups):
#             print('# of balanced %d\'s groups data : '%i, data_count[i, :])
            
#         self.filename = new_filename
#         self.attr = torch.stack(new_attr)
        return new_features


        
        

