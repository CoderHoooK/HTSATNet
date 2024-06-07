import numpy as np
from copy import deepcopy
import copy
import pickle
from torch.utils.data import Dataset
from feeders.augmentations12 import *
# from augmentations12 import *
import os 
import torch
from einops import rearrange
class Feeder(Dataset):
    def __init__(self, dir_path, pipeline, split='train'):
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
        self.dir_path = dir_path
        self.split = split
        self.data_path = os.path.join(self.dir_path,split + '_data.npy')
        if split != 'test':
            self.label_path = os.path.join(self.dir_path,split + '_label.pkl')
            self.ann = open(self.label_path,'rb')
            self.label = pickle.load(self.ann)[1]
        else: 
            self.label_path = os.path.join(self.dir_path,split + '_label.pkl')
            self.ann = open(self.label_path,'rb')
            self.label = pickle.load(self.ann)[0]
        self.data = np.load(self.data_path)        

        self.maxlen = max([i.shape[1] for i in self.data])
        self.transformers= [import_class('feeders.augmentations12'+'.'+p.pop('type'))(**p) for p in pipeline]
        self.Compose = Compose(self.transformers)
        
        
        

    def __len__(self):
        self.len = len(self.label)
        return self.len

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        x = copy.deepcopy(self.data[idx])
        x = x.squeeze(-1)
        x = rearrange(x,'c t v -> t v c')
        x = self.Compose(x)
        y = self.label[idx]
        # if self.maxlen > x.shape[0]:
        #     gap = self.maxlen - x.shape[0]
        #     zeros = np.zeros((gap,x.shape[1],x.shape[2]))
        #     x = np.concatenate((x,zeros),0)
            
        
        #train/val x:data y:label
        #test x:data y:sample name
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