# -*- coding:utf-8 -*-
"""
作者：尘小风
软件：Pycharm2020.2
"""

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,opt):
        super(Generator,self).__init__()
        ngf = opt.ngf
        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.nz,ngf*8,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf,3,kernel_size=5,stride=3,padding=1,bias=False),
            nn.Tanh())

    def forward(self,x):
        x = self.main(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator,self).__init__()
        ndf = opt.ndf
        # input :[3,96,96]
        self.main = nn.Sequential(
            nn.Conv2d(3,ndf,kernel_size=5,stride=3,padding=1,bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid())
    def forward(self,x):
        x = self.main(x).view(-1)
        return x




