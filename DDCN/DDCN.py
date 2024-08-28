import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(DilatedConvBlock, self).__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class DDCN(nn.Module):
    def __init__(self):
        super(DDCN, self).__init__()
        # Adding padding layer at the beginning to adjust the size
        self.initial_padding = nn.ReflectionPad2d(
            (1, 2, 1, 2))  # Adding 1 padding on the left, 2 on the right, 1 on the top, and 2 on the bottom

        # Dilated Block
        self.dilated_block = nn.Sequential(
            DilatedConvBlock(4, 64, 4, stride=1, dilation=1),
            DilatedConvBlock(64, 64, 4, stride=1, dilation=1),
            DilatedConvBlock(64, 64, 5, stride=1, dilation=2),
            DilatedConvBlock(64, 128, 4, stride=1, dilation=3),
            DilatedConvBlock(128, 128, 4, stride=1, dilation=4)
        )
        # Additional Layers as per the provided structure
        self.additional_layers = nn.Sequential(
            DilatedConvBlock(128, 256, 3, stride=1, dilation=5),
            DilatedConvBlock(256, 256, 3, stride=1, dilation=6),
            DilatedConvBlock(256, 192, 3, stride=1, dilation=7),
            DilatedConvBlock(192, 192, 3, stride=1, dilation=8)
        )
        self.final_conv = nn.Conv2d(192, 1, 1)  # Assuming the last layer is meant to reduce to 2 channels

    def forward(self, x):
        x = self.initial_padding(x)
        x = self.dilated_block(x)
        x = self.additional_layers(x)
        x = self.final_conv(x)
        return x
