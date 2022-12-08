from logging.config import valid_ident
from pickle import NONE
from torchvision import transforms
import pandas as pd
import PIL
import numpy as np
from torch.utils import data
import os
import torch
def getmode(mode = 'train'):
    if mode == 'train':
        img_names = pd.read_csv('Lab1_dataset/train.csv', usecols = ['names'])
        labels = pd.read_csv('Lab1_dataset/train.csv', usecols = ['label'])
        print(type(img_names))
        return np.squeeze(img_names), np.squeeze(labels)

    elif mode == 'val':
        img_names = pd.read_csv('Lab1_dataset/val.csv', usecols = ['names'])
        labels = pd.read_csv('Lab1_dataset/val.csv', usecols = ['label'])
        return np.squeeze(img_names), np.squeeze(labels)


class dataloader(data.Dataset):
    def __init__(self, root, mode = 'train', augmentation = False):
        self.root = root
        self.img_names, self.labels = getmode(mode)
        #self.img_names = torch.tensor(self.img_names)
        #self.labels = torch.tensor(self.labels)
        trans = []
        if augmentation:
            trans += augmentation
        trans.append(transforms.ToTensor())
        self.transform = transforms.Compose(trans)
        print(f'Found {len(self.img_names): 5d} images')

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_names[index])
        label = self.labels[index]
        img = PIL.Image.open(img_path)
        img = self.transform(img)
        return img, label

######try########
def tryit():
    train = dataloader(root='Lab1_dataset\\train\\train')
    train_loader = data.DataLoader(train, batch_size=8)
    print(len(train))
    for idx, (img, label) in enumerate(train_loader):
        #print('img: ', img.shape)
        print("label shape: ", label.shape)
        print("label: ", label)
        print(idx)


class testloader(data.Dataset):
    def __init__(self, root, name_list):
        self.root = root
        self.img_names = name_list

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_names[index])
        img = PIL.Image.open(img_path)
        img = transforms.Compose([transforms.ToTensor()])(img)
        return img