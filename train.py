import argparse

from model import Generator, Discriminator
from loss import GeneratorLoss
from data_utils import DatasetFromFolder

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch
import torch.nn as nn
import os

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=0, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opt = parser.parse_args()

crop_size = opt.crop_size
upscale_factor = opt.upscale_factor
batch_size = 8
num_epochs = 100
alpha = 1.1

train_set = DatasetFromFolder('data/train', crop_size, alpha=alpha)

train_loader = DataLoader(train_set, num_workers=0, batch_size=batch_size, shuffle=True)

netG = Generator(upscale_factor)
netD = Discriminator()
#netG = nn.DataParallel(netG, [2,3])
#netD = nn.DataParallel(netD, [2,3])

gen_criterion = GeneratorLoss()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    gen_criterion.cuda()

for epoch in range(1, num_epochs + 1):
    netG.train()
    netD.train()

    count = 1
    for lr_img, hr_img in train_loader:
        #print(lr_img.dtype, hr_img.shape)
        lr_img = Variable(lr_img)
        hr_img = Variable(hr_img)

        if torch.cuda.is_available():
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda()

        sr_img = netG(lr_img)

        netD.zero_grad()
        real_out = netD(hr_img).mean()
        fake_out = netD(sr_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        optimizerD.step()

        netG.zero_grad()
        g_loss = gen_criterion(fake_out, sr_img, hr_img)
        g_loss.backward()
        optimizerG.step()
        print("Training epoch {}, Batch_Num {}/{}, G_Loss {}, D_Loss {}".format(epoch, count, len(train_loader), g_loss, d_loss))
        count += 1

    # print("Training epoch {}, G_Loss {}, D_Loss {}".format(epoch, g_loss, d_loss))
    if epoch % 1 == 0:
        torch.save(netG, 'save/netG_epoch_{}_{}.pth'.format(upscale_factor, epoch))
        torch.save(netD, 'save/netD_epoch_{}_{}.pth'.format(upscale_factor, epoch))
