# -*- coding= utf-8 -*-
# @Time : 2023/3/27 11:12
# @Author : 尘小风
# @File : Model.py
# @software : PyCharm


import torchvision.models.vgg # 按住ctrl点击model.vgg即可跳转官方实现源码

import torch
from typing import cast

# 官方预训练权重
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(torch.nn.Module):
    # features是特征提取卷积层的网络结构，通过下面定义的make_features进行生成
    # init_weights表示是否对权重进行初始化
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, num_classes)
        )
        if init_weights: # 如果上面的init_weights设置成True，则通过下面定义的_initialize_weights方法进行初始化
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self): # 初始化方法
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d): # 如果是卷积层
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # (何)恺明初始化方法
                #torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: # 如果有偏置参数
                    torch.nn.init.constant_(m.bias, 0) # 把偏置参数初始化为0
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear): # 如果是全连接层
                #torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0) # 把偏置参数初始化为0


def make_layers(cfg: list, batch_norm=False):
    layers = [] # 存放网络层级
    in_channels = 3
    for v in cfg:
        if v == "M": # 如果是最大池化层
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v  # 下一层卷积层的输入是上一层的输出

    return torch.nn.Sequential(*layers) # 采用非关键字参数传入


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # VGG11层网络结构
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # # VGG13层网络结构
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # VGG16层网络结构
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # VGG19层网络结构
}


def vgg(model_name, batch_norm, **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    model = VGG(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    return model