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
    def __init__(self, data_path, pipeline, split='train',is_14=True):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """
        self.data_path = data_path
        self.ann_file = open(data_path,'rb')
        self.split = split
        self.is_14 = is_14
        self.transformers= [import_class('feeders.augmentations12'+'.'+p.pop('type'))(**p) for p in pipeline]
        self.Compose = Compose(self.transformers)
        self.load_pkl_annotations()


    def load_pkl_annotations(self):
        data = pickle.load(self.ann_file)
        if self.is_14:
            self.label = data[self.split][1]
        else: self.label = data[self.split][2]
        self.len = len(self.label)
        self.data = data[self.split][0]
        self.total_frames = self.len
        

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        x = copy.deepcopy(self.data[idx])
        x = self.Compose(x)
        y = self.label[idx]
        return x,y

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


if __name__ == '__main__':
    train_pipeline = [
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleDecode', clip_len=50),
]
    train=dict(data_path='data\DHG2016.pkl', pipeline=train_pipeline, split='train')
    dataset = Feeder(**train)
    a = dataset[0]
    b = dataset[1]