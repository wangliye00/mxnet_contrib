from __future__ import absolute_import

import mxnet
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.loss import Loss

class BinaryDiceLoss_v1(Loss):
    """Apply dice loss for binary segmentation
        DiceLoss can be customized by redefining forward and backward method

        Input:
        - pred:  prediction for groundtruth, with shape of 5D array
        - label: binary label for groundtruth, with shape of 5D array
        note: layout of 5D array - batch_size * channels * volume shape
    """

    def __init__(self, axis=1, use_softmax=True, weight=None, batch_axis=0, **kwargs):
        super(BinaryDiceLoss_v1, self).__init__(weight, batch_axis, **kwargs)
        self.axis = axis
        self.apply_softmax = use_softmax

    def hybrid_forward(self, F, pred, label, sample_weight = None):
        N = label.shape[0]

        #convert output to probability
        if self.use_softmax:
            pred = F.softmax(pred, axis = self.axis)

        pred_flat = F.Flatten(pred[:,0])
        label_flat = F.Flatten(label)

        smooth = 1e-6

        intersection = pred_flat * label_flat

        dice_coef = 2 * (F.sum(intersection)) / (F.sum(pred_flat) + F.sum(label_flat) + smooth)
        loss = 1 - dice_coef / N

        return loss

class BinaryDiceLoss_v2(mxnet.autograd.Function):

    def forward(self, pred, label, save=True, epsilon=1e-6):
        batch_size = label.shape[0]

        #conert feature map to binary label
        pred_label = nd.argmax(pred, axis=1)
        if label.dtype != pred_label.dtype:
            label = label.astype('float32')

        if save:
            self.save_for_backward(pred, label)

        pred_label = pred_label.reshape(batch_size, -1)
        target_label = label.reshape(batch_size, -1)
        self.intersect = nd.sum(pred_label * target_label, axis=1)
        input_area = nd.sum(pred_label, axis=1)
        target_area = nd.sum(target_label, axis=1)

        self.sum = input_area + target_area + 2 * epsilon

        batch_loss = 1 - 2 * self.intersect / self.sum
        loss = 0.0
        for i in range(batch_size):
            if target_area[i] != 0:
                loss += batch_loss[i]

        return loss / batch_size

    def backward(self, output_grads):
        """
        Customized backward of layers

        -----------------------------
        args:
        - output_grads : gradient w.r.t output \

        return:
        - gradient w.r.t input \
        """

        pred, target = self.saved_tensors
        intersect, sum = self.intersect, self.sum

        tmp1 = 2 / sum  
        tmp2 = 4 * intersect / sum / sum

        batch_size = a.shape[0]
        tmp1 = tmp1.reshape(batch_size, 1, 1, 1, 1)
        tmp2 = tmp2.reshape(batch_size, 1, 1, 1, 1)

        grad_diceeloss = -tmp1 * target + tmp2 * pred[:, 1:2]

        input_grads = nd.concat(grad_diceeloss * -output_grads.asscalar(),
                               grad_diceeloss * output_grads.asscalar(), dim=1)
        target_grads = nd.ones(shape=target.shape, ctx=target.context)
        
        return input_grads, target_grads


