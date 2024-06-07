import json

from torch.utils.data import Dataset
import numpy as np
import random
import numpy as np
from copy import deepcopy
import copy
import pickle
from torch.utils.data import Dataset
from feeders.augmentations12 import *
# from augmentations12 import *
import os 
import torch
class Feeder(Dataset):
    def __init__(self, data_path,pipeline,split,is_14):
        self.nw_hand17_root = '/home/hk/TD-GCN-Gesture-master/data/shrec/shrec17_jsons/'
        self.is_14 = is_14
        if 'val' in split:  
            self.train_val = 'val'
            with open(self.nw_hand17_root + 'test_samples.json', 'r') as f1:
                json_file = json.load(f1)
            self.data_dict = json_file
            self.flag = 'test_jsons/' 
        else: 
            self.train_val = 'train'
            with open(self.nw_hand17_root + 'train_samples.json', 'r') as f2:
                json_file = json.load(f2)
            self.data_dict = json_file
            self.flag = 'train_jsons/' 
        self.load_data()  
        
        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            if self.is_14:  
                self.label.append(int(info['label_14']) - 1)
            else: 
                self.label.append(int(info['label_28']) - 1)
        self.transformers= [import_class('feeders.augmentations12'+'.'+p.pop('type'))(**p) for p in pipeline]
        self.Compose = Compose(self.transformers)
    def load_data(self):
        self.data = [] # data: T N C
        for data in self.data_dict: 
            file_name = data['file_name']
            with open(self.nw_hand17_root + self.flag + file_name + '.json', 'r') as f: 
                json_file = json.load(f)
            skeletons = json_file['skeletons'] 
            value = np.array(skeletons)
            self.data.append(value)

    def __len__(self):
        return len(self.data_dict)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)] 
        value = self.data[index % len(self.data_dict)] 
        value = self.Compose(value)
        

        return value, label

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
