#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : beiqi.qh (beiqi.qh@alibaba-inc.com)
Date               : 2021-08-03 17:16:48
Last Modified By   : beiqi.qh (beiqi.qh@alibaba-inc.com)
Last Modified Date : 2021-08-05 17:45:31
Description        : multi-head 
-------- 
Copyright (c) 2021 Alibaba Inc. 
'''

import torch
import torch.nn.functional as F
from torch import nn

from fastreid.config import configurable
from fastreid.layers import *
from fastreid.layers import pooling, any_softmax
from fastreid.layers.weight_init import weights_init_kaiming
from .build import REID_HEADS_REGISTRY


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0), ) + self.shape)


def build_embedding_head(option, input_dim, output_dim, dropout_prob):
    reduce = None
    if option == 'fc':
        reduce = nn.Linear(input_dim, output_dim)
    elif option == 'dropout_fc':
        reduce = [nn.Dropout(p=dropout_prob), nn.Linear(input_dim, output_dim)]
        reduce = nn.Sequential(*reduce)
    elif option == 'bn_dropout_fc':
        reduce = [
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=dropout_prob),
            nn.Linear(input_dim, output_dim)
        ]
        reduce = nn.Sequential(*reduce)
    elif option == 'mlp':
        reduce = [
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        ]
        reduce = nn.Sequential(*reduce)
    else:
        print('unsupported embedding head options {}'.format(option))
    return reduce


class FC(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FC, self).__init__()
        self.fc = nn.Linear(inplanes, outplanes)
        #self.bn = nn.BatchNorm1d(outplanes)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        return self.act(x)


class GDN(nn.Module):
    def __init__(self, inplanes, outplanes, intermediate_dim=256):
        super(GDN, self).__init__()
        self.fc1 = FC(inplanes, intermediate_dim)
        self.fc2 = FC(intermediate_dim, outplanes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        intermediate = self.fc1(x)
        out = self.fc2(intermediate)
        # return intermediate, self.softmax(out)
        return intermediate, torch.softmax(out, dim=1)


class MultiHeads(nn.Module):
    def __init__(self,
                 feature_dim=256,
                 groups=4,
                 mode='S',
                 backbone_fc_dim=1024):
        super(MultiHeads, self).__init__()
        self.mode = mode
        self.groups = groups
        # self.Backbone = backbone[resnet]
        self.instance_fc = FC(backbone_fc_dim, feature_dim)
        self.GDN = GDN(feature_dim, groups)
        self.group_fc = nn.ModuleList(
            [FC(backbone_fc_dim, feature_dim) for i in range(groups)])
        self.feature_dim = feature_dim

    def forward(self, x):
        B = x.shape[0]
        # x = self.Backbone(x)  # (B,4096)
        instacne_representation = self.instance_fc(x)

        # GDN
        group_inter, group_prob = self.GDN(instacne_representation)
        # print(group_prob)
        # group aware repr
        v_G = [Gk(x) for Gk in self.group_fc]  # (B,512)

        # self distributed labeling
        group_label_p = group_prob.data
        group_label_E = group_label_p.mean(dim=0)
        group_label_u = (group_label_p - group_label_E.unsqueeze(
            dim=-1).expand(self.groups, B).T) / self.groups + (1 / self.groups)
        group_label = torch.argmax(group_label_u, dim=1).data

        # group ensemble
        group_mul_p_vk = list()
        if self.mode == 'S':
            for k in range(self.groups):
                Pk = group_prob[:, k].unsqueeze(dim=-1).expand(
                    B, self.feature_dim)
                group_mul_p_vk.append(torch.mul(v_G[k], Pk))
            group_ensembled = torch.stack(group_mul_p_vk).sum(dim=0)
        # instance , group aggregation
        final = instacne_representation + group_ensembled
        return group_inter, final, group_prob, group_label


@REID_HEADS_REGISTRY.register()
class MultiHead(nn.Module):
    @configurable
    def __init__(self, *, feat_dim, embedding_dim, num_classes, neck_feat,
                 pool_type, cls_type, scale, margin, with_bnneck, norm_type):
        """
        NOTE: this interface is experimental.

        Args:
            feat_dim:
            embedding_dim:
            num_classes:
            neck_feat:
            pool_type:
            cls_type:
            scale:
            margin:
            with_bnneck:
            norm_type:
        """
        super().__init__()

        # Pooling layer
        assert hasattr(pooling, pool_type), "Expected pool types are {}, " \
                                            "but got {}".format(pooling.__all__, pool_type)
        self.pool_layer = getattr(pooling, pool_type)()

        self.neck_feat = neck_feat
        self.MultiHeads = MultiHeads(feat_dim,
                                     groups=32,
                                     mode='S',
                                     backbone_fc_dim=feat_dim)

        neck = []
        if embedding_dim > 0:
            if pool_type == "Identity":
                m = nn.Linear(feat_dim * 16 * 16, embedding_dim)
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                neck.append(m)
            else:
                neck.append(nn.Linear(feat_dim, embedding_dim, bias=False))
                #neck.append(Reshape(feat_dim))
                #neck.append(nn.Linear(feat_dim, embedding_dim, bias=False))
                #neck.append(Reshape(embedding_dim, 1, 1))

            feat_dim = embedding_dim

        if with_bnneck:
            #neck.append(get_norm(norm_type, feat_dim, bias_freeze=True))
            bn = nn.BatchNorm1d(feat_dim)
            bn.bias.requires_grad_(False)
            neck.append(bn)

        self.bottleneck = nn.Sequential(*neck)

        # Classification head
        assert hasattr(any_softmax, cls_type), "Expected cls types are {}, " \
                                               "but got {}".format(any_softmax.__all__, cls_type)
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        self.cls_layer = getattr(any_softmax, cls_type)(num_classes, scale,
                                                        margin)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.bottleneck.apply(weights_init_kaiming)
        self.MultiHeads.apply(weights_init_kaiming)
        nn.init.normal_(self.weight, std=0.01)

    @classmethod
    def from_config(cls, cfg):
        # fmt: off
        feat_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        cls_type = cfg.MODEL.HEADS.CLS_LAYER
        scale = cfg.MODEL.HEADS.SCALE
        margin = cfg.MODEL.HEADS.MARGIN
        with_bnneck = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type = cfg.MODEL.HEADS.NORM
        # fmt: on
        return {
            'feat_dim': feat_dim,
            'embedding_dim': embedding_dim,
            'num_classes': num_classes,
            'neck_feat': neck_feat,
            'pool_type': pool_type,
            'cls_type': cls_type,
            'scale': scale,
            'margin': margin,
            'with_bnneck': with_bnneck,
            'norm_type': norm_type
        }

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)
        global_feat = global_feat.flatten(1)
        #multi-head
        _, global_feat, _, _ = self.MultiHeads(global_feat)
        neck_feat = self.bottleneck(global_feat)

        # Evaluation
        # fmt: off
        if not self.training: return neck_feat
        # fmt: on

        # Training
        if self.cls_layer.__class__.__name__ == 'Linear':
            logits = F.linear(neck_feat, self.weight)
        else:
            logits = F.linear(F.normalize(neck_feat), F.normalize(self.weight))

        # Pass logits.clone() into cls_layer, because there is in-place operations
        cls_outputs = self.cls_layer(logits.clone(), targets)

        # fmt: off
        if self.neck_feat == 'before': feat = global_feat
        elif self.neck_feat == 'after': feat = neck_feat
        else:
            raise KeyError(
                f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": logits.mul(self.cls_layer.s),
            "features": feat,
        }
