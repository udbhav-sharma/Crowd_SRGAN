import torch
import torch.nn as nn
from .network import Conv2d
import icc.network as network
#import torchvision
from torchvision import models
#from torch.nn.init import *

class GCEVGG16(nn.Module):

    def __init__(self, bn=False):
        super(GCEVGG16, self).__init__()



        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16 =nn.Sequential(*list(self.vgg16.features.children())[:-15])

        self.branch1 = nn.Sequential(Conv2d( 256, 196, 7, same_padding=True),
                                     Conv2d(196, 96, 5, same_padding=True),
                                     Conv2d(96, 32, 3, same_padding=True, bn=bn),
                                     Conv2d( 32, 1, 1, same_padding=True, bn=bn))

        #self.branch2 = nn.Sequential(nn.Conv2d( 32, 1, 1))


        #self.Upsampling = nn.Sequential(nn.Upsample(size=[32,32], mode='bilinear'),
                                        #Conv2d(256, 32, 3, same_padding=True, bn=bn))

        network.weights_normal_init(self.branch1, dev=0.01)
        #network.weights_normal_init(self.classifier, dev=0.01)


    def forward(self, im_data):
#         x1 = torch.cat((im_data,im_data,im_data),1)
        x1 = im_data
        return self.vgg16(x1)
