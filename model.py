import torch
import torch.nn as nn
import torch.nn.functional as F

class Upsample(nn.Module):
    ''' This layer takes input of spatial dimensions hxw and gives output [sx(h-1)+k-2p]x[sx(w-1)+k-2p] '''
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, output_padding = 1,dropout = True):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)
        return x

class Downsample(nn.Module):
    '''This layer takes input of spatial dimensions hxw and gives output [(h-k+2p)/s + 1]x[(w-k+2p)/s + 1]'''
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, apply_instancenorm = True, leaky_relu = True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=apply_instancenorm)
        self.norm = nn.InstanceNorm2d(out_channels)
        if leaky_relu:
            self.relu = nn.LeakyReLU(0.2,)
        else:
            self.relu = nn.ReLU()
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)
        return x

class Resblock(nn.Module):
    ''' Residual block with standard architecture '''	
    def __init__(self, in_channels, use_dropout = True, dropout_ratio = 0.5):
        super(Resblock, self).__init__()
        layers = list()
        # c x h x w
        layers.append(nn.ReflectionPad2d(1))
        # c x (h + 2) x (w + 2)
        layers.append(Downsample(in_channels, in_channels, 3, 1, padding = 0, leaky_relu = False))
        # c x h x w
        layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.ReflectionPad2d(1))
        # c x (h + 2) x (w + 2)
        layers.append(nn.Conv2d(in_channels, in_channels, 3, 1, padding = 0, bias = True))
        # c x h x w
        layers.append(nn.InstanceNorm2d(in_channels))
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.res(x)
    
class Generator(nn.Module):
    ''' The default Generator with nine residual blocks '''
    def __init__(self, in_channels, out_channels, num_res_blocks = 9):
        super(Generator, self).__init__()
        model = list()
        # c x h x w
        model.append(nn.ReflectionPad2d(3))
        # c x (h + 6) x (w + 6)
        model.append(Downsample(in_channels, 64, 7, 1, padding = 0, leaky_relu = False))
        # 64 x h x w
        model.append(Downsample(64, 128, 3, 2, padding = 1, leaky_relu = False))
        # 128 x floor((h + 1)/2) x floor((w + 1)/2) ~ 128 x h/2 x w/2
        model.append(Downsample(128, 256, 3, 2, padding = 1, leaky_relu = False))
        # 256 x floor((h/2 + 1)/2) x floor((w/2 + 1)/2) ~ 265 x h/4 x w/4
        for i in range(num_res_blocks):
            model.append(Resblock(256))
        # 256 x h/4 x w/4
        model.append(Upsample(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1))
        # 128 x h/2 x w/2
        model.append(Upsample(128, 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1))
        # 64 x h x w
        model.append(nn.ReflectionPad2d(3))
        # 64 x (h + 6) x (w + 6)
        model.append(nn.Conv2d(64, out_channels, kernel_size = 7, padding = 0))
        # out x h x w
        model.append(nn.Tanh())

        self.gen = nn.Sequential(*model)

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    ''' The Patch GAN classifier with default 3 layers '''
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        model = list()
        # c x h x w
        model.append(nn.Conv2d(in_channels, 64, kernel_size = 4, stride = 2, padding = 1))
        # 64 x h/2 x w/2
        model.append(nn.LeakyReLU(0.2))
        model.append(Downsample(64, 128, 4, 2, padding = 1))
        # 128 x h/4 x w/4
        model.append(Downsample(128, 256, 4, 2, padding = 1))
        # 256 x h/8 x w/8
        model.append(Downsample(256, 512, 4, 1, padding = 1))
        # 512 x (h/8 - 1) x (w/8 - 1)
        model.append(nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 1))
        # 1 x (h/8 - 2) x (w/8 - 2)
        self.dis = nn.Sequential(*model)

    def forward(self, x):
        return self.dis(x)