# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:23:40 2018

@author: liye.wang
"""

import mxnet
from mxnet import nd
from mxnet.gluon import nn

class InputConvBlock(nn.HybridBlock):
    """ First conv block for input data
        conv[3*3]+bn+relu \
    """
    
    def __init__(self, out_channels, **kwargs):
        super(InputConvBlock, self).__init__(**kwargs)
        self.ops = nn.HybridSequential()
        self.ops.add(nn.Conv3D(channels=out_channels, kernel_size=3, padding=1, use_bias=False),
                     nn.BatchNorm(),
                     nn.Activation(activation='relu')
                     )
        
    def hybrid_forward(self, F, x):
        return self.ops(x)

class ConvBlock(nn.HybridBlock):
    """ Basic conv block for DenseBlock, without bottleneck 
        bn+relu+conv[3*3] \
    """
    
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.ops = nn.HybridSequential()
        self.ops.add(nn.BatchNorm(),
                     nn.Activation(activation='relu'),
                     nn.Conv3D(channels=out_channels, kernel_size=3, padding=1, use_bias=False)
                     )
        
    def hybrid_forward(self, F, x):
        return self.ops(x)

class BottConvBlock(nn.HybridBlock):
    """ Basic conv block for DenseBlock with bottleneck
        bn+relu+conv[1*1]+bn+relu+conv[3*3] \
    """
    
    def __init__(self, growth_rate, **kwargs):
        super(BottConvBlock, self).__init__(**kwargs)
        
        bott_channels = 2 * growth_rate
        self.ops = nn.HybridSequential()
        self.ops.add(nn.BatchNorm(),
                     nn.Activation(activation='relu'),
                     nn.Conv3D(channels=bott_channels, kernel_size=1, use_bias=False),
                     nn.BatchNorm(),
                     nn.Activation(activation='relu'),
                     nn.Conv3D(channels=bott_channels, kernel_size=3, strides=1, padding=1, use_bias=False)
                     )
        
    def hybrid_forward(self, F, x):
        return self.ops(x)

class TransitionBlockDown(nn.HybridBlock):
    """ bn+relu+conv[1*1]+maxpool[2*2] 
        reduce both channels and image size by half \
    """
    
    def __init__(self, out_channels, **kwargs):
        super(TransitionBlockDown, self).__init__(**kwargs)      
        self.ops = nn.HybridSequential()
        self.ops.add(nn.BatchNorm(),
                     nn.Activation(activation='relu'),
                     nn.Conv3D(channels=out_channels, kernel_size=1, strides=1, use_bias=False),
                     nn.MaxPool3D(pool_size=2, strides=2)
                     )
        
    def hybrid_forward(self, F, x):
        return self.ops(x)
        
class TransitionBlockUp(nn.HybridBlock):
    """ bn+relu+deconv+conv[1*1] """
    
    def __init__(self, out_channels, **kwargs):
        super(TransitionBlockUp, self).__init__(**kwargs)
        self.ops = nn.HybridSequential()
        self.ops.add(nn.BatchNorm(),
                     nn.Activation(activation='relu'),
                     nn.Conv3DTranspose(channels=out_channels, kernel_size=4, strides=2, padding=1),
                     )
        
    def hybrid_forward(self, F, x):
        return self.ops(x)

class DenseBlock(nn.HybridBlock):
    """Dense convolution block"""
    
    def __init__(self, num_conv=3, growth_rate=32, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.ops = nn.HybridSequential()
        for i in range(num_conv):
            self.ops.add(BottConvBlock(growth_rate))
            
    def hybrid_forward(self, F, x):
        for layer in self.ops:
            out = layer(x)
            x = F.concat(x, out, dim=1)
        return x
    
class DownSampleBlock(nn.HybridBlock):
    """ Downsample images by half (conv[1*1]+conv[3*3])*3
    
    Parameter
    ------------------------------
    out_channels : number of output channels \
    num_layer :  number of dense block \
    """
    def __init__(self, out_channels, use_bottleneck=True, **kwargs):
        super(DownSampleBlock, self).__init__(**kwargs)
        self.ops = nn.HybridSequential()
        if use_bottleneck:
            self.ops.add(DenseBlock(),
                         TransitionBlockDown(out_channels=out_channels)
                         )
    
    def hybrid_forward(self, F, x):
        return self.ops(x)

class TransitionBlock(nn.HybridBlock):
    def __init__(self, out_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.ops = nn.HybridSequential()
        self.ops.add(nn.BatchNorm(),
                     nn.Activation(activation='relu'),
                     nn.Conv3D(channels=out_channels, kernel_size=1, use_bias=False)
                     )
    
    def hybrid_forward(self, F, x):
        return self.ops(x)

class UpSampleBlock(nn.HybridBlock):
    """"""
    def __init__(self, out_channels, use_bottleneck=True, **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.ops = nn.HybridSequential()
        if use_bottleneck:
            self.ops.add(TransitionBlockUp(out_channels=out_channels),
                         DenseBlock()
                         )
        self.rblock = TransitionBlock(out_channels=out_channels)
            
    def hybrid_forward(self, F, x, skip):
        out = self.ops(x)
        out = F.concat(out, skip, dim=1)
        return self.rblock(out)
        
        
    
class OutputConvBlock(nn.HybridBlock):
    """bn+relu+conv[3*3]"""
    
    def __init__(self, num_class, **kwargs):
        super(OutputConvBlock, self).__init__(**kwargs)
        self.ops = nn.HybridSequential()
        self.ops.add(nn.BatchNorm(),
                     nn.Activation(activation='relu'),
                     nn.Conv3D(channels=num_class, kernel_size=3, strides=1, padding=1)
                     )
    
    def hybrid_forward(self, F, x):
        return self.ops(x)
        
class BrainSegNet_v1(nn.HybridBlock):
    """Brain ROIs segmentation
    
    Members
    --------------------------
    in_channels : input channels \
    out_channels : output channels \
    layers : number of layers for DenseBlock(list) \
    DownSample : downsample block, half size \
    UpSample : upsample block, double size \
    """
    
    def __init__(self, num_class, ** kwargs):
        super(BrainSegNet_v1, self).__init__(**kwargs)
        
        self.init_channels = 256
        self.num_class = num_class
        self.ops = nn.HybridSequential()
        self.ops.add(InputConvBlock(self.init_channels),
                     DownSampleBlock(out_channels=64),
                     DownSampleBlock(out_channels=128),
                     DownSampleBlock(out_channels=256),
                     UpSampleBlock(out_channels=128),
                     UpSampleBlock(out_channels=64),
                     UpSampleBlock(out_channels=self.init_channels),
                     OutputConvBlock(num_class=self.num_class)
                     )
        
    def hybrid_forward(self, F, x):
        out_256 = self.ops[0](x)
        out_64 = self.ops[1](out_256)
        out_128 = self.ops[2](out_64)
        out = self.ops[3](out_128)
        out = self.ops[4](out, out_128)
        out = self.ops[5](out, out_64)
        out = self.ops[6](out, out_256)
        return self.ops[7](out)
        
            
        
        
        
        