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
   DiceLoss can be customized by redefining forward and backward method
    
    Inputs:
    - pred: prediction for groudtruth, 5D array
    - label: binary labels for groundtruth , 5D array
    note: 5D array - batch_size*channels*volume shape
"""
    def __init__(self, axis = 1, apply_softmax = True, weight=None, batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self.axis = axis
        self.apply_softmax = apply_softmax
    
    def hybrid_forward(self, F, pred, label, sample_weight = None):
        N = label.shape[0]
        
        #convert output to probability, and use groundtruth to calculate loss
        if apply_softmax:
            pred = F.softmax(pred, axis = self.axis)
         
        pred_falt = F.Flatten(pred[:,0])
        label_flat = F.Flatten(label)
        
        smooth = 1e-4
        
        intersection = pred_flat * label_flat
        
        dice_coef = 2 * (F.sum(intersection) + smooth) \
                    / (F.sum(pred_flat) + F.sum(label_flat) + smooth)
        loss = 1 - dice_coef / N
        
        return loss
                    
        
        
        
