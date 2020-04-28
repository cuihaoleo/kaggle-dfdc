import os
from torch.utils.data import Dataset
from strong_transform import augmentation, trans
import json
import random
import cv2


class DfdcDataset(Dataset):

    def __init__(self, datapath="", phase='train', resize=(320, 320)):
        assert phase in ['train', 'val', 'test']
        if phase == 'val':
            phase = 'valid'
        self.phase = phase
        self.resize = resize
        self.num_classes = 2
        self.epoch = 0
        self.next_epoch()
        self.aug = augmentation
        self.trans = trans
        self.datapath = datapath

    def next_epoch(self):
        with open('dfdc.json') as f:
            dfdc = json.load(f)
#        self.dataset=dfdc['train']
        if self.phase == 'train':
            trainset = dfdc['train']+dfdc['valid']
            tr = list(filter(lambda x: x[1] == 0, trainset))
            tf = random.sample(list(filter(lambda x: x[1] == 1, trainset)), len(tr))
            self.dataset = tr+tf
        if self.phase == 'valid':
            validset = dfdc['test']
            tr = list(filter(lambda x: x[1] == 0, validset))
            tf = random.sample(list(filter(lambda x: x[1] == 1, validset)), len(tr))
            self.dataset = tr+tf
        if self.phase == 'test':
            self.dataset = dfdc['test']
        self.epoch += 1

    def __getitem__(self, item):
        try:
            vid = self.dataset[item//20]
            ind = str(item % 20*12+self.epoch % 12)
            ind = '0'*(3-len(ind))+ind+'.png'
            image = cv2.imread(os.path.join(self.datapath, vid[0], ind))
            image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.resize)
            if self.phase == 'train':
                image = self.aug(image=image)['image']
            return self.trans(image), vid[1]
        except:
            return self.__getitem__((item+250) % (self.__len__()))

    def __len__(self):
        return len(self.dataset)*20
