#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:17:21 2018

@author: viresh
"""

import torch
import torch.nn as nn
from .network import Conv2d

class FCNN304(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=False):
        super(FCNN304, self).__init__()

        self.branch1 = nn.Sequential(Conv2d( 280, 196, 7, same_padding=True),
                                     Conv2d(196, 96, 5, same_padding=True),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.ReLU(),
                                     Conv2d(96, 32, 3, same_padding=True, bn=bn),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.ReLU())
                                     #Conv2d( 32, 1, 1, same_padding=True, bn=bn))
                                     #Conv2d( 24, 1, 1, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d( 32, 1, 1, same_padding=True, bn=bn))
                                     #Conv2d( 24, 1, 1, same_padding=True, bn=bn))


    def forward(self, im_data):
        x1 = self.branch1(im_data)
        return self.branch2(x1)
