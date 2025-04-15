import torch
import torch.nn as nn
import torchvision.models as models


class ResNetRegressor(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNetRegressor, self).__init__()

        # 使用预训练的ResNet
        base_model = models.resnet50(pretrained=True)
        layers = list(base_model.children())[:-2]  # 去掉最后的全连接层和平均池化层

        self.features = nn.Sequential(*layers)

        # 添加转置卷积层来上采样到期望的输出大小
        self.upconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(128, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.features(x)
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        return x