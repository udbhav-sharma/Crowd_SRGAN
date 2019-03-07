#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 22:01:21 2018

@author: vireshranjan
"""
import icc.network as network
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .model_ic_CNN import modelicCNN
import cv2
from torchvision.transforms import ToTensor

def evaluate_model(net, data_loader):
    net.eval()
    
    maeHR = 0.0
    mseHR = 0.0
    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        
        im_data = ToTensor()(im_data)
        im_data = im_data.expand(1, im_data.size()[0], im_data.size()[1], im_data.size()[2])
        
        gt_count = np.sum(gt_data)

        with torch.no_grad():
            im_data.requires_grad = False        
            im_data = im_data.cuda()
        
            HR_density_map = net(im_data)
                
            HR_density_map = HR_density_map.data.cpu().numpy()
            et_countHR = np.sum(HR_density_map)
                
            maeHR += abs(gt_count-et_countHR)
            mseHR += ((gt_count-et_countHR)*(gt_count-et_countHR))
        
    maeHR = maeHR/data_loader.get_num_samples()
    mseHR = np.sqrt(mseHR/data_loader.get_num_samples())
    return maeHR, mseHR
