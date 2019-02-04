import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from os import mkdir, listdir
from os.path import exists, join

from model import Generator

upscale_factor = 0
saved_model = 'save/netG_alpha_100.pth'

out_path = 'sr_imgs'
in_path = 'data/images'

if not exists(out_path):
    mkdir(out_path)

# model = Generator(upscale_factor)
# model.cuda()
# model.load_state_dict(torch.load('save/netG_epoch_{}_100.pth'.format(upscale_factor)))
model = torch.load(saved_model)

for img_name in listdir(in_path):
    image = Image.open(join(in_path, img_name))
    image = image.convert('RGB')
    image = Variable(ToTensor()(image), requires_grad=False).unsqueeze(0)
    image = image.cuda()

    out = model(image)
    out_img = ToPILImage()(out[0].data.cpu())

    out_img.save(join(out_path, 'out_' + img_name))
    print('Image saved {}'.format(img_name))
