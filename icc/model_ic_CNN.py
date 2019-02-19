#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 16:11:29 2018

@author: viresh
"""

import torch
import torch.nn as nn
from .modelsDMESimple import DME2 as DME
from .modelsFCNN281 import FCNN304 as FCNN352
from .modelsGCEVGG16Count import GCEVGG16
import icc.network as network

trainedDME = 'model_twoBranchSimple/mcnn_shtechA_DME_0.h5'
trainedFCNN = 'model_twoBranchSimple/mcnn_shtechA_FCNN_0.h5'
trainedGCE = 'model_twoBranchSimple/mcnn_shtechA_GCE_0.h5'

class modelicCNN(nn.Module):

    def __init__(self, bn=False):
        super(modelicCNN, self).__init__()
        self.netDME = DME(bn=bn)
        self.netFCNN = FCNN352(bn=bn)
        self.netGCE = GCEVGG16(bn=bn)


    def forward(self, im_data):
        y1 = self.netDME(im_data)
        x1 = self.netGCE(im_data) # LR prediction
        x1 = torch.cat((y1,x1),1)
        x1 = self.netFCNN(x1) # HR prediction
        return x1

class retrain_icCNN(nn.Module):
    def __init__(self, bn=False):
        super(retrain_icCNN, self).__init__()
        self.netDME = DME(bn = bn)
        network.load_net(trainedDME, self.netDME)
        self.netFCNN = FCNN352(bn = bn)
        network.load_net(trainedFCNN, self.netFCNN)
        self.netGCE = GCEVGG16(bn = bn)
        network.load_net(trainedGCE, self.netGCE)

    def forward(self, im_data):
        y1 = self.netDME(im_data)
        x1 = self.netGCE(im_data) # LR prediction
        x1 = torch.cat((y1,x1),1)
        x1 = self.netFCNN(x1) # HR prediction
        return x1
