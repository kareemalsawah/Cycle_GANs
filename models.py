import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, num_res_units):
        super().__init__()
        self.in_channels = in_channels
        self.num_res_units = num_res_units

        self.res_units = []
        for i in range(self.num_res_units):
          self.res_units.append(ResUnit(self.in_channels, filter_size=3))

        self.res_units = nn.Sequential(*self.res_units)

    def forward(self, x):
        for i in range(self.num_res_units):
          x = F.relu(self.res_units[i].forward(x))

        return x

class ResUnit(nn.Module):
    def __init__(self, num_channels, filter_size = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, filter_size, padding = 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, filter_size, padding = 1)
        self.conv1_inst_norm = nn.InstanceNorm2d(num_channels)
        self.conv2_inst_norm = nn.InstanceNorm2d(num_channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1_inst_norm(self.conv1(x)))
        x = self.conv2_inst_norm(self.conv2(x)) + residual
        return x


class Generator(nn.Module):
  def __init__(self,channel_sizes=[64,128,256],num_res_units=9):
    super().__init__()
    self.num_res_units = num_res_units
    self.channel_sizes = channel_sizes

    # Downsampling
    self.conv1 = nn.Conv2d(3,channel_sizes[0],kernel_size=7,stride=1,padding=2)
    self.conv2 = nn.Conv2d(channel_sizes[0],channel_sizes[1],kernel_size=3,stride=2,padding=1)
    self.conv3 = nn.Conv2d(channel_sizes[1],channel_sizes[2],kernel_size=3,stride=2,padding=1)
    self.conv1_inst_norm = nn.InstanceNorm2d(channel_sizes[0])
    self.conv2_inst_norm = nn.InstanceNorm2d(channel_sizes[1])
    self.conv3_inst_norm = nn.InstanceNorm2d(channel_sizes[2])

    # Resblock
    self.res_block = ResBlock(channel_sizes[2],self.num_res_units)

    # Upsampling
    self.tconv1 = nn.ConvTranspose2d(channel_sizes[2],channel_sizes[1],kernel_size=3,stride=2,padding=1,output_padding=1)
    self.tconv2 = nn.ConvTranspose2d(channel_sizes[1],channel_sizes[0],kernel_size=3,stride=2,padding=1,output_padding=1)
    self.tconv3 = nn.ConvTranspose2d(channel_sizes[0],3,kernel_size=7,stride=1,padding=3)
    self.tconv1_inst_norm = nn.InstanceNorm2d(channel_sizes[1])
    self.tconv2_inst_norm = nn.InstanceNorm2d(channel_sizes[0])

  def forward(self,x):
    x = F.relu(self.conv1_inst_norm(self.conv1(x)))
    x = F.relu(self.conv2_inst_norm(self.conv2(x)))
    x = F.relu(self.conv3_inst_norm(self.conv3(x)))

    x = F.relu(self.res_block(x))

    x = F.relu(self.tconv1_inst_norm(self.tconv1(x)))
    x = F.relu(self.tconv2_inst_norm(self.tconv2(x)))
    x = self.tconv3(x)

    return x

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1)
    self.conv1_inst_norm = nn.InstanceNorm2d(64)

    self.conv2 = nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1)
    self.conv2_inst_norm = nn.InstanceNorm2d(128)

    self.conv3 = nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1)
    self.conv3_inst_norm = nn.InstanceNorm2d(256)

    self.conv4 = nn.Conv2d(256,512,kernel_size=4,stride=1,padding=1)
    self.conv4_inst_norm = nn.InstanceNorm2d(512)

    self.conv5 = nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1)

  def forward(self,x):
    x = F.relu(self.conv1_inst_norm(self.conv1(x)))
    x = F.relu(self.conv2_inst_norm(self.conv2(x)))
    x = F.relu(self.conv3_inst_norm(self.conv3(x)))
    x = F.relu(self.conv4_inst_norm(self.conv4(x)))
    x = self.conv5(x)

    return x
