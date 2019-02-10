import torch
from torch import nn
from torchvision.models.vgg import vgg19


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = vgg.features[:28].eval()
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_labels, out_images, target_images, GT_Density, Density):
        
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        
        # Density Loss
        density_loss = self.mse_loss(GT_Density, Density)
        
        return 0.1 * (image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss) + density_loss
