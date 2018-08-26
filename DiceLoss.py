# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 12:58:50 2018

@author: wangliye
"""

from __future__ import absolute_import

from mxnet import ndarray
from mxnet import numeric_types
from mxnet.gluon import HybridBlock
from mxnet.gluon.loss import Loss

class DiceLoss(Loss):
"""Apply dice loss for binary segmentaion
    
    Inputs:
    - pred: prediction for groudtruth, 5D array
    - label: binary labels for groundtruth , 5D array
    note: 5D array - batch_size*channels*volume shape
"""
    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__(**kwargs)
    
    def hybrid_forward(self, F, pred, label, sample_weight = None):
        N = label.shape[0]
        
        pred_falt = F.Flatten(pred[:,0,:,:,:])
        label_flat = F.Flatten(label)
        
        smooth = 1e-4
        
        intersection = pred_flat * label_flat
        
        dice_coef = 2 * (F.sum(intersection, axis=1) + smooth) \
                    / (F.sum(pred_flat, axis=1) + F.sum(label_flat, axis=1) + smooth)
        loss = 1 - dice_coef
        
        return loss
                    
        
        
        