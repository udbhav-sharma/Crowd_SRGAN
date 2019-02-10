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
from .model_ic_CNN import modelicCNN
import cv2

def evaluate_model(trainedPath,data_loader,ds_flag=False):
    net = modelicCNN()
    #net = nn.DataParallel(net)
    network.load_net(trainedPath, net)
    net.cuda()
    net.eval()
    maeLR = 0.0
    mseLR = 0.0
    maeHR = 0.0
    mseHR = 0.0
    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        if ds_flag:
            orig_shape = im_data.shape
            im_data = cv2.resize(im_data, (im_data.shape[1], im_data.shape[0]))
            im_data = cv2.resize(im_data, (orig_shape[1], orig_shape[0]))
        im_data = im_data.reshape((1,1,im_data.shape[0],im_data.shape[1]))
        gt_count = np.sum(gt_data)
        im_data = torch.from_numpy(im_data).cuda()
        im_data.requires_grad = False
        LR_density_map,HR_density_map = net(im_data)
        LR_density_map = LR_density_map.data.cpu().numpy()
        et_countLR = np.sum(LR_density_map)
        HR_density_map = HR_density_map.data.cpu().numpy()
        et_countHR = np.sum(HR_density_map)
        maeLR += abs(gt_count-et_countLR)
        mseLR += ((gt_count-et_countLR)*(gt_count-et_countLR))
        maeHR += abs(gt_count-et_countHR)
        mseHR += ((gt_count-et_countHR)*(gt_count-et_countHR))
    maeLR = maeLR/data_loader.get_num_samples()
    mseLR = np.sqrt(mseLR/data_loader.get_num_samples())
    maeHR = maeHR/data_loader.get_num_samples()
    mseHR = np.sqrt(mseHR/data_loader.get_num_samples())
    return maeLR,mseLR,maeHR,mseHR