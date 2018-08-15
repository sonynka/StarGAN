# coding=utf-8

import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
from logging import getLogger
import numpy as np

logger = getLogger()

class FashionDataset(Dataset):
    def __init__(self, image_path, metadata_path, transform, attributes, mode):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.lines = pd.read_csv(metadata_path, encoding='utf-8')
        self.num_data = self.lines.shape[0]

        self.selected_attrs = [col for col in self.lines.columns
                               if any(attr in col for attr in attributes.split(','))]

        logger.info('Preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()

        if self.mode == 'train':
            self.num_data = len(self.train_filenames)
        elif self.mode == 'test':
            self.num_data = len(self.test_filenames)

    def preprocess(self):
        self.train_filenames = []
        self.train_labels = []
        self.test_filenames = []
        self.test_labels = []

        lines = self.lines[['img_path'] + self.selected_attrs]
        lines = lines.loc[(lines[self.selected_attrs]!=0).any(axis=1)]
        train_set = lines.sample(frac=0.8)
        test_set = lines.drop(train_set.index)

        self.train_filenames = train_set['img_path'].tolist()
        self.train_labels = train_set.iloc[:, 1:].as_matrix().tolist()

        self.test_filenames = test_set['img_path'].tolist()
        self.test_labels = test_set.iloc[:, 1:].as_matrix().tolist()

        logger.info('{} / {} images with for train / test sets'.format(
           len(self.train_filenames), len(self.test_filenames)))
        logger.info('Attributes: {}'.format(self.selected_attrs))


    def __getitem__(self, index):
        if self.mode == 'train':
            file_path = u''.join((self.image_path, self.train_filenames[index])).encode('utf-8').strip()
            image = Image.open(file_path)
            label = self.train_labels[index]
        elif self.mode in ['test']:
            file_path = u''.join((self.image_path, self.test_filenames[index])).encode('utf-8').strip()
            image = Image.open(file_path)
            label = self.test_labels[index]

        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_data


def default_loader(path):
    return Image.open(path).convert('RGB')

def lab_loader(path):
    from skimage import io,color

    rgb = io.imread(path)
    lab = color.rgb2lab(rgb)

    return(lab)

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageLabelFilelist(Dataset):

    def __init__(self, root, flist_path, labels_path, attributes, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):

        self.root = root
        self.labels_path = labels_path
        self.transform = transform
        self.loader = loader

        imlist = flist_reader(os.path.join(self.root, flist_path))

        labels_df = self.process_labels(attributes, imlist)
        self.imlist = labels_df.img_path.tolist()
        self.class_names = labels_df.columns[1:]
        self.labels = labels_df.iloc[:, 1:].as_matrix().tolist()


    def process_labels(self, attributes, imlist):
        """ Load the labels from CSV file, process, and return a dataframe """

        labels_df = pd.read_csv(os.path.join(self.root, self.labels_path))
        labels_cols = [
            col for col in labels_df.columns
            if any(col.startswith(attr) for attr in attributes.split(','))]

        labels_df = labels_df[['img_path'] + labels_cols]
        labels_df = labels_df.loc[(labels_df[labels_cols] != 0).any(axis=1)]
        labels_df = labels_df.loc[labels_df.img_path.isin(imlist)]
        return labels_df

    def __getitem__(self, index):

        img = self.loader(os.path.join(self.root, self.imlist[index]))
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(label)

    def __len__(self):
        return len(self.imlist)



def get_loaders(root, attributes, image_size, batch_size):
    """Build and return data loaders for all data sets"""

    modes = ['train', 'val', 'test']

    data_transforms = \
        transforms.Compose([
            transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    image_datasets = {
        mode: ImageLabelFilelist(root,
                                 mode + '_imgs.csv',
                                 'img_attr.csv',
                                 attributes,
                                 transform=data_transforms)
        for mode in modes
    }

    data_loaders = {
        mode: DataLoader(image_datasets[mode],
                         batch_size=batch_size,
                         shuffle=True if mode == 'train' else False,
                         num_workers=4)
        for mode in modes
    }

    logger.info('{train} / {test} / {val} images for train / val / test sets'\
                .format(**{mode: len(image_datasets[mode]) for mode in modes}))
    logger.info('Classes: {}'.format(image_datasets['train'].class_names))


    return data_loaders