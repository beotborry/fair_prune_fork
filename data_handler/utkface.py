from os.path import join
from PIL import Image
from utils import list_files
from natsort import natsorted
from collections import defaultdict
import random
import numpy as np
import data_handler
from torchvision import transforms
from data_handler.get_mean_std import get_mean_std

class UTKFaceDataset(data_handler.GenericDataset):
    
    label = 'age'
    sensi = 'race'
    fea_map = {
        'age' : 0,
        'gender' : 1,
        'race' : 2
    }
    num_map = {
        'age' : 100,
        'gender' : 2,
        'race' : 4
    }

    def __init__(self, img_size=224, **kwargs):
        mean, std = get_mean_std('utkface', group=kwargs['group_mode'])
        if kwargs['split'] == 'train':
            transform = transforms.Compose(
                [transforms.Resize((256, 256)),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=std)]
            )
        elif kwargs['split'] == 'test':
            transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=std)]
            )        

        super(UTKFaceDataset, self).__init__(transform=transform, **kwargs)
        
        # data pre processing
        filename = list_files(self.root, '.jpg')
        filename = natsorted(filename)
        self._data_preprocessing(filename)
        self.num_groups = self.num_map[self.sensi]
        self.num_classes = self.num_map[self.label]        
        
        random.seed(1) # we want the same train /test set, so fix the seed to 1
        random.shuffle(self.features)
        
        # split train & test
        self.features = self._make_data(self.features, self.num_groups, self.num_classes)
        self.num_data, self.idxs_per_group = self._data_count(self.features, self.num_groups, self.num_classes)
        
        # only for fairbatch... :(
        self.labels, self.groups = self._make_SY()
        
    def __getitem__(self, index):
        if self.group_mode != -1:
            index = self.idxs_per_group[self.group_mode][index]
        s, l, img_name = self.features[index]
        
        image_path = join(self.root, img_name)
        image = Image.open(image_path, mode='r').convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 1, np.float32(s), np.int64(l), (index, img_name)
    
    def _make_SY(self):
        labels = []
        groups = []
        for i in self.features:
            img_name, s_, l_ = i
            labels.append(l_)
            groups.append(s_)
        return labels, groups

    def _data_preprocessing(self, filenames):
        filenames = self._delete_incomplete_images(filenames)
        filenames = self._delete_others_n_age_filter(filenames)
        self.features = [] 
        for filename in filenames:
            s, y = self._filename2SY(filename)
            self.features.append([ s, y, filename])
    
    def _filename2SY(self, filename):        
        tmp = filename.split('_')
        sensi = int(tmp[self.fea_map[self.sensi]])
        label = int(tmp[self.fea_map[self.label]])
        if self.sensi == 'age':
            sensi = self._transform_age(sensi)
        if self.label == 'age':
            label = self._transform_age(label)
        return int(sensi), int(label)
    
    def _transform_age(self, age):
        if age<20:
            label = 0
        elif age<40:
            label = 1
        else:
            label = 2
        return label 
        
    def _delete_incomplete_images(self, filename):
        filename = [image for image in filename if len(image.split('_')) == 4]
        return filename

    def _delete_others_n_age_filter(self, filename):

        filename = [image for image in filename
                         if ((image.split('_')[self.fea_map['race']] != '4'))]
        ages = [self._transform_age(int(image.split('_')[self.fea_map['age']])) for image in filename]
        self.num_map['age'] = len(set(ages))

        return filename
