# encoding: utf-8


import os.path as osp
import random

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PKUVD(ImageDataset):
    """PKUVD.

    Train dataset statistics:
        - identities: 70591 + 39619.(VD1 + VD2)
        - images: 422326 + 342608.
    """
    dataset_dir = "PKU-VD"
    dataset_name = "pku-VD"

    def __init__(self, root='datasets', query_list='', gallery_list='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.subdir = kwargs['subdir']
        self.dataset_name = self.dataset_name + '_' + self.subdir

        self.image_dir = osp.join(self.dataset_dir, self.subdir, 'image')
        self.train_list = osp.join(
            self.dataset_dir, self.subdir, 'train_test/trainlist.txt')
        if query_list and gallery_list:
            self.query_list = query_list
            self.gallery_list = gallery_list
        else:
            self.query_list = osp.join(
                self.dataset_dir, self.subdir, 'query_ref/querylist.txt')
            self.gallery_list = osp.join(
                self.dataset_dir, self.subdir, 'query_ref/small_set.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list)
        query = self.process_dir(self.query_list, is_train=False)
        gallery = self.process_dir(self.gallery_list, is_train=False)

        super(PKUVD, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split(' ')[1])
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, f"{imgid}.jpg")
            imgid = int(imgid)
            if is_train:
                vid = f"{self.dataset_name}_{vid}"
                imgid = f"{self.dataset_name}_{imgid}"
            dataset.append((img_path, vid, imgid))
        assert len(dataset) == len(img_list_lines)
        return dataset


@DATASET_REGISTRY.register()
class PKUVD1(PKUVD):
    """pku-vd1
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        kwargs['subdir'] = 'VD1'

        super(PKUVD1, self).__init__(root,  **kwargs)


@DATASET_REGISTRY.register()
class PKUVD2(PKUVD):
    """pku-vd1
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        kwargs['subdir'] = 'VD2'

        super(PKUVD2, self).__init__(root,  **kwargs)
