import torch
import torch.nn as nn
from .network import Conv2d

class DME2(nn.Module):

    def __init__(self, bn=False):
        super(DME2, self).__init__()


        self.branch3 = nn.Sequential(Conv2d( 1, 16, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     Conv2d(48, 48, 3, same_padding=True, bn=bn),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn))

        #self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))

    def forward(self, im_data):
        x = self.branch3(im_data)
        #x = torch.cat((x1,x2,x3),1)
        #x = self.fuse(x)

        return x
