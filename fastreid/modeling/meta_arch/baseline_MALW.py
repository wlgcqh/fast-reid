# encoding: utf-8
"""
"""

import torch
from torch import nn
import numpy as np
from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class MALW_Baseline(nn.Module):
    @configurable
    def __init__(self,
                 *,
                 backbone,
                 heads,
                 pixel_mean,
                 pixel_std,
                 loss_kwargs=None):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads

        self.loss_kwargs = loss_kwargs

        self.register_buffer('pixel_mean',
                             torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

        self.ID_LOSS_NAMES = ['loss_cls', 'loss_circle', 'loss_cosface']
        self.METRIC_LOSS_NAMES = ['loss_triplet', 'loss_supcon']
        self.id_loss_history = []
        self.metric_loss_history = []
        self.update_iter_interval = 500
        self.ID_LOSS_WEIGHT = 1
        self.METRIC_LOSS_WEIGHT = 1

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs': {
                # loss name
                'loss_names': cfg.MODEL.LOSSES.NAME,

                # loss hyperparameters
                'ce': {
                    'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                    'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                    'scale': cfg.MODEL.LOSSES.CE.SCALE
                },
                'tri': {
                    'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                    'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                    'scale': cfg.MODEL.LOSSES.TRI.SCALE
                },
                'supcon': {
                    'num_ids':
                    cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                    'views': cfg.DATALOADER.NUM_INSTANCE,
                    'scale': cfg.MODEL.LOSSES.SUPCON.SCALE
                },
                'circle': {
                    'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                    'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                    'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                },
                'cosface': {
                    'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                    'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                    'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                }
            }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError(
                "batched_inputs must be dict or torch.Tensor, but get {}".
                format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs, gt_labels, ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')) * ce_kwargs.get('scale')
            id_loss = loss_dict['loss_cls']

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features, gt_labels, tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')) * tri_kwargs.get('scale')
            metric_loss = loss_dict['loss_triplet']

        if 'SupContrastLoss' in loss_names:
            supcon_kwargs = self.loss_kwargs.get('supcon')
            loss_dict['loss_supcon'] = supcontrast_loss(
                pred_features,
                gt_labels,
                num_ids=supcon_kwargs.get('num_ids'),
                views=supcon_kwargs.get('views')) * supcon_kwargs.get('scale')
            metric_loss = loss_dict['loss_supcon']

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features, gt_labels, circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')) * circle_kwargs.get('scale')
            id_loss = loss_dict['loss_circle']

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')
            id_loss = loss_dict['loss_cosface']

        self.id_loss_history.append(id_loss.item())
        self.metric_loss_history.append(metric_loss.item())
        if len(self.id_loss_history) == 0:
            pass
        elif (len(self.id_loss_history) % self.update_iter_interval == 0):
            id_history = np.array(self.id_loss_history)
            id_mean = id_history.mean()
            id_std = id_history.std()

            metric_history = np.array(self.metric_loss_history)
            metric_mean = metric_history.mean()
            metric_std = metric_history.std()

            id_weighted = id_std
            metric_weighted = metric_std
            if id_weighted > metric_weighted:
                new_weight = 1 - (id_weighted - metric_weighted) / id_weighted
                self.ID_LOSS_WEIGHT = self.ID_LOSS_WEIGHT * 0.9 + new_weight * 0.1

            self.id_loss_history = []
            self.metric_loss_history = []
            print(
                f"update weighted loss ID_LOSS_WEIGHT={round(self.ID_LOSS_WEIGHT,3)},METRIC_LOSS_WEIGHT={self.METRIC_LOSS_WEIGHT}"
            )

        else:
            pass

        for loss_name, loss_value in loss_dict.items():
            if loss_name in self.ID_LOSS_NAMES:
                loss_dict[loss_name] *= self.ID_LOSS_WEIGHT
            elif loss_name in self.METRIC_LOSS_NAMES:
                loss_dict[loss_name] *= self.METRIC_LOSS_WEIGHT
            else:
                raise Exception(f"unknown loss name: {loss_name}")
        return loss_dict
