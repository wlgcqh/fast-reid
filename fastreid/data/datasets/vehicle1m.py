# encoding: utf-8
'''
File: vehicle1m.py
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
class Vehicle1M(ImageDataset):
    """Vehicle1M.

    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.

    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_

    Train dataset statistics:
        - identities: 50000.
        - images: 844571.
    """
    dataset_dir = "Vehicle-1M"
    dataset_name = "vehicle1m"

    def __init__(self, root='datasets', test_list='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = osp.join(
            self.dataset_dir, 'train-test-split/train_list.txt')
        # self.vehicle_info = osp.join(
        #     self.dataset_dir, 'train_test_split/vehicle_info.txt')
        if test_list:
            self.test_list = test_list
        else:
            self.test_list = osp.join(
                self.dataset_dir, 'train-test-split/test_3000.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(Vehicle1M, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split(' ')[1])
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, imgid)
            imgid = int(imgid.split('/')[1].split('.')[0])
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


@DATASET_REGISTRY.register()
class SmallVehicle1M(Vehicle1M):
    """VehicleID.
    Small test dataset statistics:
        - identities: 1000.
        - images: 16123.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(
            dataset_dir, 'train-test-split/test_1000.txt')

        super(SmallVehicle1M, self).__init__(root, self.test_list, **kwargs)


@DATASET_REGISTRY.register()
class MediumVehicle1M(Vehicle1M):
    """VehicleID.
    Medium test dataset statistics:
        - identities: 2000.
        - images: 32539.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(
            dataset_dir, 'train-test-split/test_2000.txt')

        super(MediumVehicle1M, self).__init__(root, self.test_list, **kwargs)


@DATASET_REGISTRY.register()
class LargeVehicle1M(Vehicle1M):
    """VehicleID.
    Large test dataset statistics:
        - identities: 3000.
        - images: 49259.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(
            dataset_dir, 'train-test-split/test_3000.txt')

        super(LargeVehicle1M, self).__init__(root, self.test_list, **kwargs)
