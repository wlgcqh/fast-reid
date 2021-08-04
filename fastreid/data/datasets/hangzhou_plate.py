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
class HangzhouPlate(ImageDataset):
    """HangzhouPlate.

    Train dataset statistics:
        - identities: 125798.
        - images: 11532681.
    Test dataset statistics:
        - identities: 1500.
        - images: 1401676.
    """
    dataset_dir = "hangzhou_plate"
    dataset_name = "hangzhou_plate"

    def __init__(self, root='datasets', test_list='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'all_pass')
        self.train_list = osp.join(
            self.dataset_dir, 'train_test_split/train.txt')

        self.test_list = osp.join(
            self.dataset_dir, 'train_test_split/test.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(HangzhouPlate, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            img_path = osp.join(self.dataset_dir, line.split(' ')[0])
            vid = int(line.split(' ')[1])
            imgid = int(line.split(' ')[2])
            if is_train:
                vid = f"{self.dataset_name}_{vid}"
                imgid = f"{self.dataset_name}_{imgid}"
            dataset.append((img_path, vid, imgid))

        if is_train:
            return dataset
        else:
            random.shuffle(dataset)
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
