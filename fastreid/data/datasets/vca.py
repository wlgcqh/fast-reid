# encoding: utf-8
'''
File: HangzhouPlate.py
Project: datasets
File Created: Thursday, 17th June 2021 4:34:50 pm
Author: 北齐 (beiqi.qh@alibaba-inc.com)
-----
Last Modified: Thursday, 17th June 2021 4:35:11 pm
Modified By: 北齐 (beiqi.qh@alibaba-inc.com>)
-----
Copyright 2021 Alibaba Group AIC.
'''

import os.path as osp
import random
from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VCA(ImageDataset):

    dataset_dir = "vca"
    train_names = ['luzhou', 'fuzhou', 'hangzhou']
    test_name = 'quanzhou'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_lists = [
            osp.join(self.dataset_dir, f'{name}.txt')
            for name in self.train_names
        ]

        self.test_list = osp.join(self.dataset_dir, f'{self.test_name}.txt')

        required_files = [
            self.dataset_dir,
            *self.train_lists,
            self.test_list,
        ]
        self.check_before_run(required_files)
        train = []
        for dataset_name, train_list in zip(self.train_names,
                                            self.train_lists):
            train.extend(
                self.process_dir(train_list, dataset_name, is_train=True))

        query, gallery = self.process_dir(self.test_list,
                                          self.test_name,
                                          is_train=False)

        super(VCA, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, dataset_name, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            img_path = osp.join(self.dataset_dir, line.split(',')[0])
            vid = int(line.split(',')[1])
            # imgid = int(line.split(',')[2])
            imgid = idx
            if is_train:
                vid = f"{dataset_name}_{vid}"
                imgid = f"{dataset_name}_{imgid}"
            dataset.append((img_path, vid, imgid))

        if is_train:
            return dataset
        else:
            #random.shuffle(dataset)
            vid_container = set()
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    query.append(sample)
                else:
                    gallery.append(sample)

            return query, gallery
