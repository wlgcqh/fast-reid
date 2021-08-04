# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import REID_HEADS_REGISTRY, build_heads

# import all the meta_arch, so they will be registered
from .embedding_head import EmbeddingHead
from .multi_head import MultiHead
from .clas_head import ClasHead
from .reduce_head import ReduceHead
