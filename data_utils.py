from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, ToPILImage


def is_image(fname):
    return any(fname.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, alpha):
    return Compose([
        ToPILImage(),
        Resize(int(crop_size // alpha), interpolation=Image.BICUBIC),
        Resize(crop_size, interpolation=Image.BICUBIC),
        ToTensor()
    ])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, alpha=1.1):
        super(DatasetFromFolder, self).__init__()
        self.images = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image(x)]
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, alpha)
        self.crop_size = crop_size

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        img = img.convert('RGB')
        width, height = img.size

        # Resizing image such that width and height are multiples of crop_size
        width += self.crop_size - (width % self.crop_size)
        height += self.crop_size - (height % self.crop_size)
        img = img.resize((width, height))

        hr_image = self.hr_transform(img)
        lr_image = self.lr_transform(hr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.images)
