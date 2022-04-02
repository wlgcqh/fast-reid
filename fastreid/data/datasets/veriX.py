# encoding: utf-8
"""
@author:  Jinkai Zheng
@contact: 1315673509@qq.com
"""

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class VeRiX(ImageDataset):
    """veri for vehicleX.

    Dataset statistics:
        - identities: 1362.
        - images: 75,516.
    """
    dataset_dir = "veriX"
    dataset_name = "veriX"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'VeRi_ReID_Simulation')

        required_files = [self.dataset_dir, self.train_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = []
        gallery = []

        super(VeRiX, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid
            assert 1 <= camid
            camid -= 1  # index starts from 0

            pid = self.dataset_name + "_" + str(pid)
            camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
