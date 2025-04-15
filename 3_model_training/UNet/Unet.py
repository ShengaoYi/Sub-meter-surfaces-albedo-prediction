import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder = nn.ModuleList([
            conv_block(input_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512)
        ])

        self.pool = nn.MaxPool2d(2)
        self.middle = conv_block(512, 1024)
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, 2, stride=2),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ConvTranspose2d(128, 64, 2, stride=2)
        ])
        self.decoder = nn.ModuleList([
            conv_block(1024, 512),
            conv_block(512, 256),
            conv_block(256, 128),
            conv_block(128, 64)
        ])
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        enc_features = []
        for encoder in self.encoder:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)
        x = self.middle(x)
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder)):
            x = upconv(x)
            x = torch.cat([x, enc_features[-i-1]], dim=1)
            x = decoder(x)
        x = self.final(x)
        return x
